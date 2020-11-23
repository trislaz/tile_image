#!/usr/bin/env nextflow
glob_wsi = '/gpfsdsstore/projects/rech/gdg/uub32zv/data_sfp/train/raw/*.tif' // insert glob pattern at the end (*.tiff for instance)
root_out = '/gpfsscratch/rech/gdg/uub32zv/working_data/moco_29_10'
path_mask = '/gpfsdsstore/projects/rech/gdg/uub32zv/data_sfp/train/masks/auto'
model_path = '/gpfsdswork/projects/rech/gdg/uub32zv/experiments/models/checkpoint_0127.pth.tar'
level = 2 
mask_level = -4
size = 256
auto_mask = 1
tiler = 'moco' // dispo : simple | imagenet | imagenet_v2 | moco
dataset = Channel.fromPath(glob_wsi)
				 .map { file -> tuple(file.baseName, file) } 
				 .into { dataset_1; dataset_2}
root_outputs = file("${root_out}/${tiler}/size_${size}/res_${level}/")
process Tiling {
	publishDir "$root_outputs/mat/", overwrite:true, pattern:"*_embedded.npy", mode: 'copy'
	publishDir "$root_outputs/visu/", overwrite: true, pattern: "*.png", mode: 'copy'
	publishDir "$root_outputs/info/", overwrite:true, pattern: "*_infomat.npy", mode: 'copy'
	publishDir "$root_outputs/info/", overwrite:true, pattern: "*.pickle", mode: 'copy'
	publishDir "$root_outputs/info/", overwrite:true, pattern: "*.csv", mode: 'copy'

	queue "gpu_p1"
    clusterOptions "--gres=gpu:1"
	time { 30.m * task.attempt }
    errorStrategy 'ignore'
	maxRetries 4

	input:
	set val(slideID), val(slidePath) from dataset_1

	output:
	val slideID into out
	file('*.csv')
	file('*.npy')
	file('*.pickle')
	file('*.png')

	script:
	output_folder = file("$root_outputs/${slideID}")
	python_script = file("../scripts/main_tiling.py")
	"""
	module load cuda/10.0
	python ${python_script} --path_wsi ${slidePath} \
							--path_mask ${path_mask} \
							--level $level \
							--auto_mask ${auto_mask} \
							--tiler ${tiler} \
							--size $size \
							--mask_level ${mask_level} \
                            --model_path ${model_path}
	"""
}


// Normaliser et PCAiser: à faire dans un autre script NF car cela va dépendre systematiquement du dataset à utiliser.
// Exemple : si je veux utiliser le mélance de TCGA et Curie, il faudra que je normalise sur cet ensemble.

out .into{ out_1; out_2}
out_1 .collect()
	  .into{all_wsi_tiled_1; all_wsi_tiled_2}

//process ComputeGlobalMean {
//	publishDir "${output_folder}", overwrite: true, mode: 'copy'
//	queue "prepost"
//	memory { 10.GB }
//
//	input:
//	val _ from all_wsi_tiled_1
//
//	output:
//	file('mean.npy')
//
//	script:
//
//	compute_mean = file('../scripts/compute_mean_pca.py')
//	mat_folder = "${root_outputs}/mat/"
//	output_folder = "${root_outputs}/mean/"
//	"""
//	python $compute_mean --path ${mat_folder}
//	"""
//}
//
process Incremental_PCA {
	publishDir "${output_folder}", overwrite: true, pattern: "*.txt", mode: 'copy'
	publishDir "${output_folder}", overwrite:true, pattern: "*.joblib", mode: 'copy'
	queue 'prepost'
	memory '60GB'
	cpus '4'

	input:
	val _ from all_wsi_tiled_2
	
	output:
	file("*.joblib") into results_PCA
	file("*.txt")

	script:
	output_folder = file("${root_outputs}/pca/")
	mat_folder = "${root_outputs}"
	python_script = file("../scripts/pca_partial.py")
	"""
	python $python_script --path ${mat_folder} --tiler ${tiler}
	"""
}


process Transform_Tiles {
	publishDir "${output_folder}", overwrite: true, mode: 'copy'
	queue "prepost"
	memory '20GB'

	input:
	file pca from results_PCA

	output:
	file("*.npy") into transform_tiles

	script:
	output_folder = file("${root_outputs}/mat_pca/")
	path_encoded =  "${root_outputs}"
	python_script = file("../scripts/transform_tile.py")
	"""
	python ${python_script} --path ${path_encoded}
	"""
}
