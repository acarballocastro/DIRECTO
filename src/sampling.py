import numpy as np

def calculate_mean_std_per_key(dictionaries):
    # Initialize a dictionary to store the mean and std for each key
    result = {}

    # Get all the keys from the first dictionary (assuming all dictionaries have the same keys)
    keys = dictionaries[0].keys()

    # Loop through each key and calculate the mean and std
    for key in keys:
        # Extract the values for this key from each dictionary
        values = [d[key] for d in dictionaries]
        
        # Calculate the mean and standard deviation for the current key
        mean = np.mean(values)
        std_dev = np.std(values)
        
        # Store the result in the result dictionary
        result[key] = {'mean': mean, 'std': std_dev}
    
    return result

dict1 = {'test/out_degree': 0.6373603991002141, 'test/in_degree': 0.635990601897851, 'test/clustering': 1.099421238558441, 'test/spectre': 0.6450674680882835, 'test/wavelet': 0.5764576304141005, 'test/er_acc': 0.0, 'test/dag_acc': 1.0, 'test/dag_er_acc': 0.0, 'test/frac_unique': 1.0, 'test/frac_unique_non_iso': 1.0, 'test/frac_unic_non_iso_valid': 0.0, 'test/frac_non_iso': 1.0, 'test/frac_non_unique_non_validated': 0.0, 'test/frac_isomorphic_non_validated': 0.0, 'ratio/average_ratio': -1}
dict2 = {'test/out_degree': 0.593583473316128, 'test/in_degree': 0.6141231902390545, 'test/clustering': 1.0506712385583814, 'test/spectre': 0.5749359953049618, 'test/wavelet': 0.49533257568410183, 'test/er_acc': 0.0, 'test/dag_acc': 1.0, 'test/dag_er_acc': 0.0, 'test/frac_unique': 1.0, 'test/frac_unique_non_iso': 1.0, 'test/frac_unic_non_iso_valid': 0.0, 'test/frac_non_iso': 1.0, 'test/frac_non_unique_non_validated': 0.0, 'test/frac_isomorphic_non_validated': 0.0, 'test/frac_isomorphic_non_validated2': 0.0, 'ratio/test/out_degree_ratio': 52.52951091293169, 'ratio/test/in_degree_ratio': 59.62361070282082, 'ratio/test/clustering_ratio': 29.596372917137508, 'ratio/test/spectre_ratio': 151.29894613288468, 'ratio/test/wavelet_ratio': 206.3885732017091, 'ratio/average_ratio': 99.88740277349676}
dict3 = {'test/out_degree': 0.625387466903315, 'test/in_degree': 0.6359140205227365, 'test/clustering': 1.0506712385584611, 'test/spectre': 0.6583194854877359, 'test/wavelet': 0.5989413247578137, 'test/er_acc': 0.0, 'test/dag_acc': 1.0, 'test/dag_er_acc': 0.0, 'test/frac_unique': 1.0, 'test/frac_unique_non_iso': 1.0, 'test/frac_unic_non_iso_valid': 0.0, 'test/frac_non_iso': 1.0, 'test/frac_non_unique_non_validated': 0.0, 'test/frac_isomorphic_non_validated': 0.0, 'test/frac_isomorphic_non_validated2': 0.0, 'ratio/test/out_degree_ratio': 55.344023619762396, 'ratio/test/in_degree_ratio': 61.73922529346956, 'ratio/test/clustering_ratio': 29.596372917139753, 'ratio/test/spectre_ratio': 173.24196986519365, 'ratio/test/wavelet_ratio': 249.55888531575573, 'ratio/average_ratio': 113.8960954022642}
dict4 = {'test/out_degree': 0.6210888325642081, 'test/in_degree': 0.6251617911803011, 'test/clustering': 1.050671238558459, 'test/spectre': 0.6109432914317336, 'test/wavelet': 0.5330952282280963, 'test/er_acc': 0.0, 'test/dag_acc': 1.0, 'test/dag_er_acc': 0.0, 'test/frac_unique': 1.0, 'test/frac_unique_non_iso': 1.0, 'test/frac_unic_non_iso_valid': 0.0, 'test/frac_non_iso': 1.0, 'test/frac_non_unique_non_validated': 0.0, 'test/frac_isomorphic_non_validated': 0.0, 'test/frac_isomorphic_non_validated2': 0.0, 'ratio/test/out_degree_ratio': 54.963613501257356, 'ratio/test/in_degree_ratio': 60.69531953206807, 'ratio/test/clustering_ratio': 29.59637291713969, 'ratio/test/spectre_ratio': 160.77455037677203, 'ratio/test/wavelet_ratio': 222.1230117617068, 'ratio/average_ratio': 105.63057361778878}
dict5 = {'test/out_degree': 0.6016679998717414, 'test/in_degree': 0.6117241513850802, 'test/clustering': 1.0031851255525452, 'test/spectre': 0.590963316357735, 'test/wavelet': 0.5122594069143231, 'test/er_acc': 0.0, 'test/dag_acc': 1.0, 'test/dag_er_acc': 0.0, 'test/frac_unique': 1.0, 'test/frac_unique_non_iso': 1.0, 'test/frac_unic_non_iso_valid': 0.0, 'test/frac_non_iso': 1.0, 'test/frac_non_unique_non_validated': 0.0, 'test/frac_isomorphic_non_validated': 0.0, 'test/frac_isomorphic_non_validated2': 0.0, 'ratio/test/out_degree_ratio': 53.24495574086207, 'ratio/test/in_degree_ratio': 59.39069430923109, 'ratio/test/clustering_ratio': 28.258735931057615, 'ratio/test/spectre_ratio': 155.51666219940395, 'ratio/test/wavelet_ratio': 213.44141954763467, 'ratio/average_ratio': 101.97049354563788}

# dict1 = {'test/out_degree': 0.019063224889753982, 'test/in_degree': 0.0009393963995052435, 'test/clustering': 0.058008261992001584, 'test/spectre': 0.020807614716919165, 'test/wavelet': 0.0031189197097938326, 'test/ba_acc': 1.0, 'test/dag_acc': 0.5, 'test/dag_ba_acc': 0.5, 'test/frac_unique': 1.0, 'test/frac_unique_non_iso': 1.0, 'test/frac_unic_non_iso_valid': 0.5, 'test/frac_non_iso': 1.0, 'test/frac_non_unique_non_validated': 0.0, 'test/frac_isomorphic_non_validated': 0.0, 'test/frac_isomorphic_non_validated2': 0.0, 'ratio/average_ratio': -1}
# dict2 = {'test/out_degree': 0.017592821611695175, 'test/in_degree': 0.0008781457893651812, 'test/clustering': 0.05203789472795562, 'test/spectre': 0.01845542263039257, 'test/wavelet': 0.004709243005784458, 'test/ba_acc': 1.0, 'test/dag_acc': 0.45, 'test/dag_ba_acc': 0.45, 'test/frac_unique': 1.0, 'test/frac_unique_non_iso': 1.0, 'test/frac_unic_non_iso_valid': 0.45, 'test/frac_non_iso': 1.0, 'test/frac_non_unique_non_validated': 0.0, 'test/frac_isomorphic_non_validated': 0.0, 'test/frac_isomorphic_non_validated2': 0.0, 'ratio/average_ratio': -1}
# dict3 = {'test/out_degree': 0.01914251510563214, 'test/in_degree': 0.0008788865096513998, 'test/clustering': 0.05803046719672493, 'test/spectre': 0.02186785442032191, 'test/wavelet': 0.00432064623165096, 'test/ba_acc': 1.0, 'test/dag_acc': 0.5, 'test/dag_ba_acc': 0.5, 'test/frac_unique': 1.0, 'test/frac_unique_non_iso': 1.0, 'test/frac_unic_non_iso_valid': 0.5, 'test/frac_non_iso': 1.0, 'test/frac_non_unique_non_validated': 0.0, 'test/frac_isomorphic_non_validated': 0.0, 'test/frac_isomorphic_non_validated2': 0.0, 'ratio/average_ratio': -1}
# dict4 = {'test/out_degree': 0.01681780974420133, 'test/in_degree': 0.0012112566338167152, 'test/clustering': 0.05589296468218661, 'test/spectre': 0.012173361950384765, 'test/wavelet': 0.002804078485213246, 'test/ba_acc': 1.0, 'test/dag_acc': 0.4, 'test/dag_ba_acc': 0.4, 'test/frac_unique': 1.0, 'test/frac_unique_non_iso': 1.0, 'test/frac_unic_non_iso_valid': 0.4, 'test/frac_non_iso': 1.0, 'test/frac_non_unique_non_validated': 0.0, 'test/frac_isomorphic_non_validated': 0.0, 'test/frac_isomorphic_non_validated2': 0.0, 'ratio/average_ratio': -1}
# dict5 = {'test/out_degree': 0.019549139072171595, 'test/in_degree': 0.0012122674328560734, 'test/clustering': 0.05351675183718665, 'test/spectre': 0.015887245693898322, 'test/wavelet': 0.003605512795853638, 'test/ba_acc': 1.0, 'test/dag_acc': 0.425, 'test/dag_ba_acc': 0.425, 'test/frac_unique': 1.0, 'test/frac_unique_non_iso': 1.0, 'test/frac_unic_non_iso_valid': 0.425, 'test/frac_non_iso': 1.0, 'test/frac_non_unique_non_validated': 0.0, 'test/frac_isomorphic_non_validated': 0.0, 'test/frac_isomorphic_non_validated2': 0.0, 'ratio/average_ratio': -1}

# Call the function with all the dictionaries in a list
dictionaries = [dict1, dict2, dict3, dict4, dict5]
result = calculate_mean_std_per_key(dictionaries)

# Print the result
for key, stats in result.items():
    print(f"{key}: Mean = {stats['mean']}, Std = {stats['std']}")

def compute_ratios(gen_metrics, ref_metrics, metrics_keys):
    print("Computing ratios of metrics: ", metrics_keys)
    if ref_metrics is not None and len(metrics_keys) > 0:
        ratios = {}
        for key in metrics_keys:
            try:
                ref_metric = round(ref_metrics[key], 4)
            except:
                print(key, "not found")
                continue
            if ref_metric != 0.0:
                ratios["ratio/" + key + "_ratio"] = gen_metrics[key] / ref_metric
            else:
                print(f"WARNING: Reference {key} is 0. Skipping its ratio.")
        if len(ratios) > 0:
            ratios["ratio/average_ratio"] = sum(ratios.values()) / len(ratios)
        else:
            ratios["ratio/average_ratio"] = -1
            print(f"WARNING: no ratio being saved.")
    else:
        print("WARNING: No reference metrics for ratio computation.")
        ratios = {}
    return ratios

train = {'test/out_degree': 0.011273234427319423, 'test/in_degree': 0.010329423603096854, 'test/clustering': 0.03554586555946579, 'test/spectre': 0.003803267965213575, 'test/wavelet': 0.0023589566319239808, 'test/er_acc': 0.9921875, 'test/dag_acc': 1.0, 'test/dag_er_acc': 0.9921875, 'test/frac_unique': 1.0, 'test/frac_unique_non_iso': 0.0, 'test/frac_unic_non_iso_valid': 0.0, 'test/frac_non_iso': 0.0}

ratio1 = compute_ratios(dict1, train, ['test/out_degree', 'test/in_degree', 'test/clustering', 'test/spectre', 'test/wavelet'])
ratio2 = compute_ratios(dict2, train, ['test/out_degree', 'test/in_degree', 'test/clustering', 'test/spectre', 'test/wavelet'])
ratio3 = compute_ratios(dict3, train, ['test/out_degree', 'test/in_degree', 'test/clustering', 'test/spectre', 'test/wavelet'])
ratio4 = compute_ratios(dict4, train, ['test/out_degree', 'test/in_degree', 'test/clustering', 'test/spectre', 'test/wavelet'])
ratio5 = compute_ratios(dict5, train, ['test/out_degree', 'test/in_degree', 'test/clustering', 'test/spectre', 'test/wavelet'])

print(ratio1)

# Mean and std of ratios
ratios = [ratio1['ratio/average_ratio'], ratio2['ratio/average_ratio'], ratio3['ratio/average_ratio'], ratio4['ratio/average_ratio'], ratio5['ratio/average_ratio']]
ratios_mean = np.mean(ratios, axis=0)
ratios_std = np.std(ratios, axis=0)
print("Ratios mean and std: ", ratios_mean)
print("Ratios std: ", ratios_std)