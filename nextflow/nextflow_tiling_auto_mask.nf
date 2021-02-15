#!/usr/bin/env nextflow

// Nb : ne jamais finir un path par / dans les paramètres d'entrée.

// Ne modifier que les paramètres de ce préambule

// ------------------------------------------------------------------------------

ext = "ndpi"
input_path = "/gpfsstore/rech/gdg/uub32zv/essai"
root_out = "/gpfsstore/rech/gdg/uub32zv/essai_tiled"
level_sampling = 1
size = 224 
tiler = 'imagenet'  // dispo : simple | imagenet | moco
model_path = '/path/to/moco/model' 
clusteropt_gpu = '--account=gdg@gpu --gres=gpu:1'
clusteropt_cpu = '--account=gdg@gpu -p prepost'

// ------------------------------------------------------------------------------ 

input_wsi = Channel.from([input_path])
level_mask = -1

process Mask{
    publishDir "${root_out}/masks", overwrite: true, mode: 'copy'
    clusterOptions clusteropt_cpu
	memory '40GB'
    time '30min'


    input:
    val path from input_wsi

    output:
    file('*.npy')
    val(1) into masked

    script:
    python_script = file("../scripts/make_masks.py")
    """
    python ${python_script} --path ${path} --level_mask ${level_mask}
    """
}

masked .last() .set{mask_done}

auto_mask = 1
glob_wsi = "${input_path}/*.${ext}"
path_mask = "${root_out}/masks"
dataset = Channel.fromPath(glob_wsi)
				 .map { file -> tuple(file.baseName, file) } 
				 .into { dataset_1; dataset_2}
dataset_1 .combine(mask_done)
            .set{dataset_rdy}
root_outputs = file("${root_out}/${tiler}/size_${size}/res_${level_sampling}/")
process Tiling {
	publishDir "$root_outputs/mat/", overwrite:true, pattern:"*_embedded.npy", mode: 'copy'
	publishDir "$root_outputs/visu/", overwrite: true, pattern: "*.png", mode: 'copy'
	publishDir "$root_outputs/info/", overwrite:true, pattern: "*_infomat.npy", mode: 'copy'
	publishDir "$root_outputs/info/", overwrite:true, pattern: "*.pickle", mode: 'copy'
	publishDir "$root_outputs/info/", overwrite:true, pattern: "*.csv", mode: 'copy'

    clusterOptions clusteropt_gpu
	time { 30.m * task.attempt }
    errorStrategy 'ignore'
	maxRetries 4

	input:
	set val(slideID), val(slidePath), val(_) from dataset_rdy

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
							--level $level_sampling \
							--auto_mask ${auto_mask} \
							--tiler ${tiler} \
							--size $size \
							--mask_level ${level_mask} \
                            --model_path ${model_path}
	"""
}


out .into{ out_1; out_2}
out_1 .collect()
	  .into{all_wsi_tiled_1; all_wsi_tiled_2}

process Incremental_PCA {
	publishDir "${output_folder}", overwrite: true, pattern: "*.txt", mode: 'copy'
	publishDir "${output_folder}", overwrite:true, pattern: "*.joblib", mode: 'copy'
    clusterOptions clusteropt_cpu
    
	memory '60GB'
    time '30min'
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
    clusterOptions clusteropt_cpu
	memory '40GB'
    time '30min'

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
