#!/usr/bin/env nextflow

glob_wsi = '/mnt/data4/tlazard/data/tcga/tcga_all/images/*.svs' // insert glob pattern at the end (*.tiff for instance)
root_out = '/mnt/data4/tlazard/data/tcga/tcga_all/tiled/256/res_0'
path_mask = '/mnt/data4/tlazard/data/tcga/tcga_all/annotations/auto'
level = 0 
mask_level = -1
size = 256 
auto_mask = 1
tiler = 'simple' // dispo : simple | imagenet | imagenet_v2
dataset = Channel.fromPath(glob_wsi)
				 .map { file -> tuple(file.baseName, file) } 
				 .into { dataset_1; dataset_2}
root_outputs = file("${root_out}/${tiler}/size_${size}/res_${level}/")

process Tiling {
	if (tiler == 'simple' or tiler == 'imagenet_v2'){
		publishDir "${output_folder}", overwrite: true, pattern: "*.jpg", mode: 'copy'
		publishDir "${output_folder}", overwrite: true, pattern: "tile_*.npy", mode: 'copy'
	}
	else {
		publishDir "$root_outputs/mat/", overwrite:true, pattern:"*_embedded.npy", mode: 'copy'
	}
	publishDir "$root_outputs/visu/", overwrite: true, pattern: "*.png", mode: 'copy'
	publishDir "$root_outputs/info/", overwrite:true, pattern: "*_infomat.npy", mode: 'copy'
	publishDir "$root_outputs/info/", overwrite:true, pattern: "*.pickle", mode: 'copy'
	publishDir "$root_outputs/info/", overwrite:true, pattern: "*.csv", mode: 'copy'

	queue "gpu-cbio"
	maxForks 10
	if (tiler == 'simple'){
    clusterOptions "--gres=gpu:1"
	}
    memory '20GB'

	input:
	set val(slideID), file(slidePath) from dataset_1

	output:
	val slideID into out
	file('*.csv')
	file('*.npy')
	file('*.pickle')
	file('*.jpg')
	file('*.png')

	script:
	slideID = slidePath.baseName
	output_folder = "$root_outputs/${slideID}"
	python_script = file("../scripts/main_tiling.py")
	"""
	module load cuda10.0
	python ${python_script} --path_wsi ${slidePath} \
							--path_mask ${path_mask} \
							--level $level \
							--auto_mask ${auto_mask} \
							--tiler ${tiler} \
							--size $size \
							--mask_level ${mask_level} 
	"""
}

// Normaliser et PCAiser: à faire dans un autre script NF car cela va dépendre systematiquement du dataset à utiliser.
// Exemple : si je veux utiliser le mélance de TCGA et Curie, il faudra que je normalise sur cet ensemble.

out .into{ out_1; out_2}
out_1 .collect()
	  .into{all_wsi_tiled_1; all_wsi_tiled_2}
//
//if (tiler != 'simple'){
//	process ComputeGlobalMean {
//		publishDir "${output_folder}", overwrite: true, mode: 'copy'
//		memory { 10.GB }
//
//		input:
//		val _ from all_wsi_tiled_1
//
//		output:
//		file('mean.npy')
//
//		script:
//		compute_mean = file('../scripts/compute_mean.py')
//		mat_folder = "${root_outputs}/mat/"
//		output_folder = "${root_outputs}/mean/"
//		"""
//		python $compute_mean --path ${results_folder}
//		"""
//	}



	process Incremental_PCA {
    publishDir "${output_folder}", overwrite: true, pattern: "*.txt", mode: 'copy'
	publishDir "${output_folder}", overwrite:true, pattern: "*.joblib", mode: 'copy'
    memory '60GB'
    cpus '4'

    input:
    val _ from all_wsi_tiled_2
    
    output:
    file("*.joblib") into results_PCA
	file("*.txt")

    script:
    output_folder = "${root_outputs}/pca/"
	mat_folder = "${root_outputs}"
    python_script = file("../scripts/pca_partial.py")
    """
    python $python_script --path ${mat_folder} --tiler ${tiler}
    """
	}

	input_transform = results_PCA . combine(out_2)

	process Transform_Tiles {
    publishDir "${output_folder}", overwrite: true, mode: 'copy'
    memory '20GB'

    input:
    set file(pca), val(slideID) from input_transform

    output:
    file("*.npy") into transform_tiles

    script:
	output_folder = "${root_outputs}/mat_pca/"
	path_encoded =  "$root_outputs/mat/${slideID}_embedded.npy"
    python_script = file("../scripts/transform_tile.py")
    """
    python ${python_script} --path ${path_encoded} --pca $pca
    """
	}
}
