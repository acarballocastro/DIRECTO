import os
import pathlib
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import InMemoryDataset, download_url
from collections import Counter, defaultdict
from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos

import requests, zipfile, io
import json
from tqdm import tqdm
import networkx as nx
import torch_geometric
import torch_geometric.utils as pyg_utils


class VisualGenomeDataset(InMemoryDataset):
    """
    Scene Graph Dataset for Visual Genome.
    This dataset is a collection of scene graphs extracted from the Visual Genome dataset.
    """

    def __init__(
        self, split, cfg, root, transform=None, pre_transform=None, pre_filter=None
    ):
        self.cfg = cfg
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [
            "object_alias.txt",
            "relationship_alias.txt",
            "objects.json",
            "relationships.json",
            "attributes.json",
            "vg_splits.json",
        ]

    @property
    def processed_file_names(self):
        return [f"{self.split}.pt"]

    def download(self):
        """
        Download raw data files.
        """

        # URLs for the Visual Genome dataset
        urls = [
            (
                "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects.json.zip",
                True,
            ),
            (
                "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationships.json.zip",
                True,
            ),
            (
                "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/object_alias.txt",
                False,
            ),
            (
                "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationship_alias.txt",
                False,
            ),
            (
                "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/attributes.json.zip",
                True,
            ),
            (
                "https://raw.githubusercontent.com/google/sg2im/master/sg2im/data/vg_splits.json",
                False,
            ),
        ]

        for url, is_zip in urls:
            # Extract filename from URL
            filename = url.split("/")[-1]
            filepath = os.path.join(self.raw_dir, filename)

            # Download if not already present
            if not os.path.exists(filepath):
                print(f"Downloading {filename}...")
                response = requests.get(url)
                response.raise_for_status()
                with open(filepath, "wb") as f:
                    f.write(response.content)
            else:
                print(f"Already exists: {filename}")

            # Extract if zip
            if is_zip and filename.endswith(".zip"):
                print(f"Extracting {filename}...")
                with zipfile.ZipFile(filepath, "r") as zip_ref:
                    zip_ref.extractall(self.raw_dir)
                os.remove(filepath)  # Remove the zip file after extraction

    def load_alias_map(self, alias_path):
        aliases = {}
        print('Loading aliases from "%s"' % alias_path)
        with open(alias_path, "r") as f:
            for line in f:
                line = [s.strip() for s in line.split(",")]
                for s in line:
                    aliases[s] = line[0]
        return aliases

    def create_object_vocab(
        self, min_object_instances, image_ids, objects, aliases, vocab
    ):
        """Create a vocabulary of objects from the training set."""

        # Create vocabulary of objects from the image ids (split ids)
        image_ids = set(image_ids)
        print("Making object vocab from %d training images" % len(image_ids))
        object_name_counter = Counter()
        for image in objects:
            if image["image_id"] not in image_ids:
                continue
            for obj in image["objects"]:
                names = set()
                for name in obj["names"]:
                    names.add(aliases.get(name, name))
                object_name_counter.update(names)

        # Keep only objects with >= min_object_instances
        object_names = ["__image__"]
        for name, count in object_name_counter.most_common():
            if count >= min_object_instances:
                object_names.append(name)
        print(
            "Found %d object categories with >= %d training instances"
            % (len(object_names), min_object_instances)
        )

        # Create mappings between object names and indices
        object_name_to_idx = {}
        object_idx_to_name = []
        for idx, name in enumerate(object_names):
            object_name_to_idx[name] = idx
            object_idx_to_name.append(name)

        vocab["object_name_to_idx"] = object_name_to_idx
        vocab["object_idx_to_name"] = object_idx_to_name

    def create_attribute_vocab(
        self, min_attribute_instances, image_ids, attributes, vocab
    ):
        """Create a vocabulary of attributes from the training set."""

        # Create vocabulary of attributes from the image ids (split ids)
        image_ids = set(image_ids)
        print("Making attribute vocab from %d training images" % len(image_ids))
        attribute_name_counter = Counter()
        for image in attributes:
            if image["image_id"] not in image_ids:
                continue
            for attribute in image["attributes"]:
                names = set()
                try:
                    for name in attribute["attributes"]:
                        names.add(name)
                    attribute_name_counter.update(names)
                except KeyError:
                    pass

        # Keep only attributes with >= min_attribute_instances
        attribute_names = []
        for name, count in attribute_name_counter.most_common():
            if count >= min_attribute_instances:
                attribute_names.append(name)
        print(
            "Found %d attribute categories with >= %d training instances"
            % (len(attribute_names), min_attribute_instances)
        )

        # Create mappings between attribute names and indices
        attribute_name_to_idx = {}
        attribute_idx_to_name = []
        for idx, name in enumerate(attribute_names):
            attribute_name_to_idx[name] = idx
            attribute_idx_to_name.append(name)
        vocab["attribute_name_to_idx"] = attribute_name_to_idx
        vocab["attribute_idx_to_name"] = attribute_idx_to_name

    def filter_objects(self, min_object_size, objects, aliases, vocab, splits):
        """Filter objects based on size (objects that are too small in the image are ignored) and create a mapping from object IDs to objects."""
        # Gather image ids from all splits
        all_image_ids = set()
        for image_ids in splits.values():
            all_image_ids |= set(image_ids)

        object_name_to_idx = vocab["object_name_to_idx"]
        object_id_to_obj = {}

        num_too_small = 0
        for image in objects:
            image_id = image["image_id"]
            if image_id not in all_image_ids:
                continue
            for obj in image["objects"]:
                object_id = obj["object_id"]
                final_name = None
                final_name_idx = None
                for name in obj["names"]:
                    name = aliases.get(name, name)
                    if name in object_name_to_idx:
                        final_name = name
                        final_name_idx = object_name_to_idx[final_name]
                        break
                w, h = obj["w"], obj["h"]
                # filter out objects that are too small
                too_small = (w < min_object_size) or (h < min_object_size)
                if too_small:
                    num_too_small += 1
                if final_name is not None and not too_small:
                    object_id_to_obj[object_id] = {
                        "name": final_name,
                        "name_idx": final_name_idx,
                        "box": [obj["x"], obj["y"], obj["w"], obj["h"]],
                    }
            print(
                "Skipped %d objects with size < %d" % (num_too_small, min_object_size)
            )

        return object_id_to_obj

    def create_rel_vocab(
        self,
        min_relationship_instances,
        image_ids,
        relationships,
        object_id_to_obj,
        rel_aliases,
        vocab,
    ):
        """Create a vocabulary of relationships from the training set."""

        # Create vocabulary of relationships from the image ids (split ids)
        pred_counter = defaultdict(int)
        image_ids_set = set(image_ids)
        for image in relationships:
            image_id = image["image_id"]
            if image_id not in image_ids_set:
                continue
            for rel in image["relationships"]:
                sid = rel["subject"]["object_id"]
                oid = rel["object"]["object_id"]
                found_subject = sid in object_id_to_obj
                found_object = oid in object_id_to_obj
                if not found_subject or not found_object:
                    continue
                pred = rel["predicate"].lower().strip()
                pred = rel_aliases.get(pred, pred)
                rel["predicate"] = pred
                pred_counter[pred] += 1

        # Keep only relationships with >= min_relationship_instances
        pred_names = ["__in_image__"]
        for pred, count in pred_counter.items():
            if count >= min_relationship_instances:
                pred_names.append(pred)
        print(
            "Found %d relationship types with >= %d training instances"
            % (len(pred_names), min_relationship_instances)
        )

        # Create mappings between relationship names and indices
        pred_name_to_idx = {}
        pred_idx_to_name = []
        for idx, name in enumerate(pred_names):
            pred_name_to_idx[name] = idx
            pred_idx_to_name.append(name)

        vocab["pred_name_to_idx"] = pred_name_to_idx
        vocab["pred_idx_to_name"] = pred_idx_to_name

    def encode_graphs(
        self, cfg, splits, objects, relationships, vocab, object_id_to_obj, attributes
    ):
        """
        Encode the graphs into numpy arrays for each split.
        Each graph is represented as a dictionary with the following keys:
        - image_ids: List of image IDs
        - object_ids: List of object IDs
        - object_names: List of object names (indices)
        - object_boxes: List of object bounding boxes
        - objects_per_image: List of number of objects per image
        - relationship_ids: List of relationship IDs
        - relationship_subjects: List of subject object IDs (indices)
        - relationship_predicates: List of relationship predicates (indices)
        - relationship_objects: List of object IDs (indices)
        - relationships_per_image: List of number of relationships per image
        - attributes_per_object: List of number of attributes per object
        - object_attributes: List of object attributes (indices)
        """

        # Create a mapping from object IDs to object names and boxes
        image_id_to_objects = {}
        for image in objects:
            image_id = image["image_id"]
            image_id_to_objects[image_id] = image["objects"]
        image_id_to_relationships = {}
        for image in relationships:
            image_id = image["image_id"]
            image_id_to_relationships[image_id] = image["relationships"]
        image_id_to_attributes = {}
        for image in attributes:
            image_id = image["image_id"]
            image_id_to_attributes[image_id] = image["attributes"]

        # Create numpy arrays for each split
        # We need to filters out objects and relationships that are not in the vocab, images that have too many or too few objects or relationships
        numpy_arrays = {}
        for split, image_ids in splits.items():
            skip_stats = defaultdict(int)
            # We need to filter *again* based on number of objects and relationships
            final_image_ids = []
            object_ids = []
            object_names = []
            object_boxes = []
            objects_per_image = []
            relationship_ids = []
            relationship_subjects = []
            relationship_predicates = []
            relationship_objects = []
            relationships_per_image = []
            attribute_ids = []
            attributes_per_object = []
            object_attributes = []
            for image_id in image_ids:
                image_object_ids = []
                image_object_names = []
                image_object_boxes = []
                object_id_to_idx = {}
                for obj in image_id_to_objects[image_id]:
                    object_id = obj["object_id"]
                    if object_id not in object_id_to_obj:
                        continue
                    obj = object_id_to_obj[object_id]
                    object_id_to_idx[object_id] = len(image_object_ids)
                    image_object_ids.append(object_id)
                    image_object_names.append(obj["name_idx"])
                    image_object_boxes.append(obj["box"])
                num_objects = len(image_object_ids)
                too_few = num_objects < cfg.min_objects_per_image
                too_many = num_objects > cfg.max_objects_per_image
                if too_few:
                    skip_stats["too_few_objects"] += 1
                    continue
                if too_many:
                    skip_stats["too_many_objects"] += 1
                    continue
                image_rel_ids = []
                image_rel_subs = []
                image_rel_preds = []
                image_rel_objs = []
                for rel in image_id_to_relationships[image_id]:
                    relationship_id = rel["relationship_id"]
                    pred = rel["predicate"]
                    pred_idx = vocab["pred_name_to_idx"].get(pred, None)
                    if pred_idx is None:
                        continue
                    sid = rel["subject"]["object_id"]
                    sidx = object_id_to_idx.get(sid, None)
                    oid = rel["object"]["object_id"]
                    oidx = object_id_to_idx.get(oid, None)
                    if sidx is None or oidx is None:
                        continue
                    image_rel_ids.append(relationship_id)
                    image_rel_subs.append(sidx)
                    image_rel_preds.append(pred_idx)
                    image_rel_objs.append(oidx)
                num_relationships = len(image_rel_ids)
                too_few = num_relationships < cfg.min_relationships_per_image
                too_many = num_relationships > cfg.max_relationships_per_image
                if too_few:
                    skip_stats["too_few_relationships"] += 1
                    continue
                if too_many:
                    skip_stats["too_many_relationships"] += 1
                    continue

                # Get attributes for each object
                obj_id_to_attributes = {}
                num_attributes = []
                for obj_attribute in image_id_to_attributes[image_id]:
                    obj_id_to_attributes[obj_attribute["object_id"]] = (
                        obj_attribute.get("attributes", None)
                    )
                for object_id in image_object_ids:
                    attributes = obj_id_to_attributes.get(object_id, None)
                    if attributes is None:
                        object_attributes.append([-1] * cfg.max_attributes_per_image)
                        num_attributes.append(0)
                    else:
                        attribute_ids = []
                        for attribute in attributes:
                            if attribute in vocab["attribute_name_to_idx"]:
                                attribute_ids.append(
                                    vocab["attribute_name_to_idx"][attribute]
                                )
                            if len(attribute_ids) >= cfg.max_attributes_per_image:
                                break
                        num_attributes.append(len(attribute_ids))
                        pad_len = cfg.max_attributes_per_image - len(attribute_ids)
                        attribute_ids = attribute_ids + [-1] * pad_len
                        object_attributes.append(attribute_ids)

                # Pad object info out to max_objects_per_image
                while len(image_object_ids) < cfg.max_objects_per_image:
                    image_object_ids.append(-1)
                    image_object_names.append(-1)
                    image_object_boxes.append([-1, -1, -1, -1])
                    num_attributes.append(-1)

                # Pad relationship info out to max_relationships_per_image
                while len(image_rel_ids) < cfg.max_relationships_per_image:
                    image_rel_ids.append(-1)
                    image_rel_subs.append(-1)
                    image_rel_preds.append(-1)
                    image_rel_objs.append(-1)

                final_image_ids.append(image_id)
                object_ids.append(image_object_ids)
                object_names.append(image_object_names)
                object_boxes.append(image_object_boxes)
                objects_per_image.append(num_objects)
                relationship_ids.append(image_rel_ids)
                relationship_subjects.append(image_rel_subs)
                relationship_predicates.append(image_rel_preds)
                relationship_objects.append(image_rel_objs)
                relationships_per_image.append(num_relationships)
                attributes_per_object.append(num_attributes)

            print('Skip stats for split "%s"' % split)
            for stat, count in skip_stats.items():
                print(stat, count)
            print()
            numpy_arrays[split] = {
                "image_ids": np.asarray(final_image_ids),
                "object_ids": np.asarray(object_ids),
                "object_names": np.asarray(object_names),
                "object_boxes": np.asarray(object_boxes),
                "objects_per_image": np.asarray(objects_per_image),
                "relationship_ids": np.asarray(relationship_ids),
                "relationship_subjects": np.asarray(relationship_subjects),
                "relationship_predicates": np.asarray(relationship_predicates),
                "relationship_objects": np.asarray(relationship_objects),
                "relationships_per_image": np.asarray(relationships_per_image),
                "attributes_per_object": np.asarray(attributes_per_object),
                "object_attributes": np.asarray(object_attributes),
            }
            for k, v in numpy_arrays[split].items():
                if v.dtype == np.int64:
                    numpy_arrays[split][k] = v.astype(np.int32)
        return numpy_arrays

    def process(self):
        """
        Processes raw Visual Genome datasets into a PyTorch Geometric-compatible format.
        """

        print(f"Processing {self.split} dataset...")

        # Loading dictionary of synonyms
        object_alias_map = self.load_alias_map(self.raw_paths[0])
        relationship_alias_map = self.load_alias_map(self.raw_paths[1])

        # Loading objects and relationships
        with open(self.raw_paths[2], "r") as f:
            objects_data = json.load(f)
        with open(self.raw_paths[3], "r") as f:
            relationships_data = json.load(f)
        with open(self.raw_paths[4], "r") as f:
            attributes_data = json.load(f)

        im_id_to_info = {}

        # Relationships
        rel_counter = Counter()
        for image in tqdm(relationships_data, desc="Processing relationships data"):
            image_id = image["image_id"]
            relationships = image["relationships"]
            im_id_to_info[image_id] = {"relationships": {}}
            for relationship in relationships:
                # Store the information in a dictionary
                canonical_predicate = relationship_alias_map.get(
                    relationship["predicate"], relationship["predicate"]
                )
                im_id_to_info[image_id]["relationships"][
                    relationship["relationship_id"]
                ] = {
                    "predicate": canonical_predicate,
                    "subject": relationship["subject"]["object_id"],
                    "object": relationship["object"]["object_id"],
                }
                rel_counter[relationship["predicate"]] += 1

        # Objects
        obj_counter = Counter()
        for image in tqdm(objects_data, desc="Processing objects data"):
            image_id = image["image_id"]
            objects = image["objects"]
            im_id_to_info[image_id]["objects"] = {}
            for obj in objects:
                obj_name = obj["names"][0]
                # canonicalize the object name
                canonical_obj = object_alias_map.get(obj_name, obj_name)
                im_id_to_info[image_id]["objects"][obj["object_id"]] = canonical_obj
                obj_counter[canonical_obj] += 1

                # Double check if there are multiple names for the same object_id
                if len(obj["names"]) > 1:
                    print(
                        f"More than one name for object_id {obj['object_id']}: {obj['names']}"
                    )
                    raise ValueError(
                        f"More than one name for object_id {obj['object_id']}: {obj['names']}"
                    )

        # Attributes
        attr_counter = Counter()
        for image in tqdm(attributes_data, desc="Processing attributes data"):
            image_id = image["image_id"]
            attributes = image["attributes"]
            im_id_to_info[image_id]["attributes"] = {}
            for attr in attributes:
                object_id = attr["object_id"]
                object_attributes = attr["attributes"] if "attributes" in attr else []
                if object_id not in im_id_to_info[image_id]["attributes"]:
                    im_id_to_info[image_id]["attributes"][object_id] = set()
                im_id_to_info[image_id]["attributes"][object_id].update(
                    object_attributes
                )
                for obj_attr in object_attributes:
                    attr_counter[obj_attr] += 1

        # Create vocabularies for top K objects, relationships, and attributes
        topK_objects = [obj for obj, _ in obj_counter.most_common(self.cfg.num_objects)]
        topK_predicates = [
            rel for rel, _ in rel_counter.most_common(self.cfg.num_relationships)
        ]
        topK_attributes = [
            attr for attr, _ in attr_counter.most_common(self.cfg.num_attributes)
        ]

        # Create mappings between object names and indices
        obj_token_to_idx = {obj: idx for idx, obj in enumerate(topK_objects)}
        pred_token_to_idx = {
            pred: idx + len(topK_objects) for idx, pred in enumerate(topK_predicates)
        }
        attr_token_to_idx = {
            attr: idx + len(topK_objects) + len(topK_predicates)
            for idx, attr in enumerate(topK_attributes)
        }

        nx_graphs = []
        for image_id, image_info in tqdm(
            im_id_to_info.items(),
            desc="Processing scene graphs to organized dictionary",
        ):
            # Create a new directed graph for each image
            G = nx.DiGraph()

            # Add nodes for objects
            added_object_ids = []
            for obj_id, obj_name in image_info["objects"].items():
                label = obj_token_to_idx.get(obj_name, -1)
                if label != -1:
                    G.add_node(obj_id, label=label)
                    added_object_ids.append(obj_id)

            # Add nodes for attributes
            for obj_id, attrs in image_info["attributes"].items():
                # First condition: check if the object is in the topK objects
                # Second condition: check if the object name (there are some attributes that have object id that is not existent in the objects... 🤔)
                if obj_id in added_object_ids:
                    for attr in attrs:
                        label = attr_token_to_idx.get(attr, -1)
                        if label != -1:
                            attr_node_id = f"{obj_id}_{attr}"
                            G.add_node(attr_node_id, label=label)
                            G.add_edge(obj_id, attr_node_id)

            # Add edges for relationships
            for rel_id, rel_info in image_info["relationships"].items():
                subject_id = rel_info["subject"]
                obj_id = rel_info["object"]
                if (subject_id in added_object_ids) and (obj_id in added_object_ids):
                    predicate = rel_info["predicate"]
                    rel_node_id = f"{subject_id}_{predicate}_{obj_id}"

                    label = pred_token_to_idx.get(predicate, -1)
                    if label != -1:
                        G.add_node(rel_node_id, label=label)
                        G.add_edge(subject_id, rel_node_id)
                        G.add_edge(rel_node_id, obj_id)

            nx_graphs.append(G)

        # Filter out graphs that are too small or too large and that have too few edges
        nx_graphs = self.filter_out_graphs(nx_graphs)

        # Convert to PyTorch Geometric format
        tg_graphs = self.convert_nx_to_tg(nx_graphs)

        # randomly split the graphs into train, val, and test sets
        num_graphs = len(tg_graphs)
        test_size = round(num_graphs * 0.2)
        train_size = round((num_graphs - test_size) * 0.8)
        val_size = num_graphs - train_size - test_size
        print(f"Dataset sizes: train {train_size}, val {val_size}, test {test_size}")

        # assign to different splits
        g_cpu = torch.Generator()
        g_cpu.manual_seed(0)
        indices = torch.randperm(num_graphs, generator=g_cpu)
        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]
        test_indices = indices[train_size + val_size :]

        # Build splits
        train_data_list = []
        val_data_list = []
        test_data_list = []
        for tg_graph_idx, tg_graph in enumerate(
            tqdm(tg_graphs, desc="Building splits")
        ):
            if tg_graph_idx in train_indices:
                train_data_list.append(tg_graph)
            elif tg_graph_idx in val_indices:
                val_data_list.append(tg_graph)
            elif tg_graph_idx in test_indices:
                test_data_list.append(tg_graph)
            else:
                raise ValueError(f"Index {tg_graph} not in any split")

        torch.save(self.collate(train_data_list), self.processed_paths[0])
        torch.save(self.collate(val_data_list), self.processed_paths[1])
        torch.save(self.collate(test_data_list), self.processed_paths[2])

    def filter_out_graphs(self, graphs):
        """
        Filter out graphs that are too small or too large and that have too few edges.
        """

        min_nodes = self.cfg.min_nodes_per_graph
        max_nodes = self.cfg.max_nodes_per_graph
        min_edges = self.cfg.min_edges_per_graph

        filtered_graphs = []
        for graph in tqdm(graphs, desc="Filtering graphs"):
            num_nodes = graph.number_of_nodes()
            number_of_edges = graph.number_of_edges()
            if min_nodes <= num_nodes <= max_nodes and number_of_edges >= min_edges:
                filtered_graphs.append(graph)

        print(f"Filtered out {len(graphs) - len(filtered_graphs)} graphs")
        print(f"Keeping {len(filtered_graphs)} graphs")
        # Some stats
        # number of nodes
        max_nodes = max(len(G.nodes) for G in filtered_graphs)
        mean_nodes = sum(len(G.nodes) for G in filtered_graphs) / len(filtered_graphs)
        min_nodes = min(len(G.nodes) for G in filtered_graphs)
        print("Max number of nodes:", max_nodes)
        print("Mean number of nodes:", mean_nodes)
        print("Min number of nodes:", min_nodes)

        # number of edges
        max_edges = max(len(G.edges) for G in filtered_graphs)
        mean_edges = sum(len(G.edges) for G in filtered_graphs) / len(filtered_graphs)
        min_edges = min(len(G.edges) for G in filtered_graphs)
        print("Max number of edges:", max_edges)
        print("Mean number of edges:", mean_edges)
        print("Min number of edges:", min_edges)

        # Check acyclicity
        num_cyclic_graphs = 0
        for graph in filtered_graphs:
            if not nx.is_directed_acyclic_graph(graph):
                num_cyclic_graphs += 1

        print(f"Number of cyclic graphs: {num_cyclic_graphs}")

        return filtered_graphs

    def convert_nx_to_tg(self, nx_graphs):
        """
        Convert NetworkX graphs to PyTorch Geometric format.
        """

        tg_graphs = []
        for nx_graph in tqdm(nx_graphs, desc="Converting to PyTorch Geometric format"):
            tg_graph = pyg_utils.from_networkx(nx_graph)
            edge_index = tg_graph.edge_index
            labels = tg_graph.label
            num_classes = (
                self.cfg.num_objects
                + self.cfg.num_relationships
                + self.cfg.num_attributes
            )
            x = F.one_hot(labels, num_classes=num_classes).float()
            # No edge attributes so setting them all to 1
            edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
            edge_attr[:, 1] = 1
            y = torch.zeros([1, 0]).float()

            data = torch_geometric.data.Data(
                x=x, edge_index=edge_index, edge_attr=edge_attr, y=y
            )
            tg_graphs.append(data)

            if self.pre_transform is not None:
                data = self.pre_transform(data)

        return tg_graphs


class VisualGenomeDataModule(AbstractDataModule):
    """
    Data module for Visual Genome datasets compatible with PyTorch Geometric.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)

        datasets = {
            "train": VisualGenomeDataset(
                split="train", cfg=cfg.dataset, root=root_path
            ),
            "val": VisualGenomeDataset(split="val", cfg=cfg.dataset, root=root_path),
            "test": VisualGenomeDataset(split="test", cfg=cfg.dataset, root=root_path),
        }

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]


class VisualGenomeDatasetInfos(AbstractDatasetInfos):
    """
    Metadata and information about the Visual Genome Dataset.
    """

    def __init__(self, datamodule, dataset_config):
        # self.datamodule = datamodule
        self.name = "visual_genome"
        self.n_nodes = datamodule.node_counts()
        self.node_types = datamodule.node_types()
        self.edge_types = datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)
