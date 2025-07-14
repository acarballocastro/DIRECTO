# Directo: Generating Directed Graphs with Dual Attention and Asymmetric Encoding

  A Pytorch implementation to generated directed graphs in the discrete flow matching framework.
  
  Paper: https://arxiv.org/abs/2506.16404

## 💻 Environment installation

There are two options for environment installation: a **conda environment** or using **pixi**.

🐍 To create the conda environment: 
```bash
conda env create -f environment.yaml
conda activate directo
```
✨ To use pixi, install it and run the desired command preceeded by:
```bash
pixi run ...
```

## 👟 Training
  
The script is runnable from `main.py` (inside [`src`](src/)) which uses Hydra overrides.
    
### Debug mode

```bash
python main.py +experiment=debug
```

### Full training
An example of a training run would be:

  ```bash 
  python main.py +experiment=visual_genome dataset=visual_genome dataset.acyclic=False model.extra_features=rrwp-ppr
  ```

### ⚙️ Parameters

The following **parameters** can be configured:

- **`dataset`, `experiment`**:  
  Choose one of:  `synthetic` | `tpu_tile` | `visual_genome`

- **`model.extra_features`**:  
  These are the **positional encodings** it is possible to choose from.
  Extra features to include:  `null` | `eigenvalues` | `magnetic-eigenvalues` | `rrwp` | `rrwp-ppr` | `mult-magnetic-eigenvalues` (Q = 5) | `mult10-magnetic-eigenvalues` (Q = 10)

- **`dataset.acyclic`**:  
  Set to `true` or `false` depending on the dataset (only relevant for some synthetic datasets, otherwise the default is already set in the corresponding Hydra config).

Additional options are available for the `synthetic` dataset. If you select `dataset=synthetic`, you can further configure:

- **`dataset.graph_type`**:  
  Graph generation model:  `er` | `sbm` | `ba`

- **`dataset.acyclic`**:  
  Whether to generate directed acyclic graphs: `true` (for ER, BA) | `false` (for ER, SBM)

To run **discrete diffusion** set ```model.backbone=diffusion```

## 🎰 Sampling

### Sampling optimization

To search for the best sampling configuration, you can run:

```bash
python main.py +experiment=visual_genome dataset=visual_genome dataset.acyclic=False \
    model.extra_features=rrwp-ppr general.num_sample_fold=5 general.test_only=your_checkpoint \
    sample.search=all
```

⚙️ Available parameters

* `all` — searches through the three parameters
* `stochasticity` — searches over stochastic level
* `distortion` — searches over distortion functions
* `target_guidance` — searches over targuet guidance levels


 ### ✅ Final sampling

Once the best sampling parameters are identified, update the configuration accordingly and run:

```bash
python main.py +experiment=visual_genome dataset=visual_genome dataset.acyclic=False \
    model.extra_features=rrwp-ppr general.num_sample_fold=5 general.test_only=your_checkpoint \
    sample.eta=10 sample.omega=0.1 sample.time_distortion=polydec
```

## 🧪 Baselines

Baseline experiments follow a similar procedure and their adaptation can be found in the [`src/baselines`](src/baselines) directory.

## 📚 Citation

```bibtex
@article{carballo2025directo,
  title     = {Generating Directed Graphs with Dual Attention and Asymmetric Encoding},
  author    = {Carballo-Castro, Alba and Madeira, Manuel and Qin, Yiming and Thanou, Dorina and Frossard, Pascal},
  year      = {2025},
  url       = {https://arxiv.org/abs/2506.16404}
}
```