#!/usr/bin/env nextflow

glob_wsi = '' // insert glob pattern at the end (*.tiff for instance)
path_mask = ''
level = 1 
size = 256 
path_mask = "/mnt/data3/pnaylor/Data/Biopsy_Nature_3-0/tissue_segmentation/"
auto_mask = 0
tiler = 'imagenet' // dispo : simple | imagenet 
python_script = "../scripts/main_tiling.py"
dataset = Channel.fromPath(slides_folder)
				 .map { file -> tuple(file.baseName, file) }
root_outputs = file("/mnt/data4/tcga/tiled/${tiler}/size_${size}/res_${level}/")

process Tiling {
	publishDir "$root_outputs/${slideID}_info", overwrite: true, pattern: "*.pickle", mode: 'copy'
	publishDir "$root_outputs/${slideID}_info", overwrite: true, pattern: "*.csv", mode: 'copy'
	publishDir "$root_outputs/${slideID}_info", overwrite: true, pattern: "infomat.npy", mode: 'copy'  
	publishDir "$root_outputs/${slideID}", overwrite: true, pattern: "*.jpg", mode: 'copy'
	publishDir "$root_outputs/visu/", overwrite: true, pattern: "*.png", mode: 'copy'
	publishDir "$root_outputs/mat/", overwrite:true, pattern:"*_embedded.npy", mode: 'copy'
	queue 'cpu'
	stageInMode 'copy'
	memory '10GB'
	
	input:
	set slideID, file(slidePath) from dataset
	each level from levels

	output:
	val slideID into out
	file('*.csv')
	file('*.npy')
	file('*.pickle')
	file('*.png')

	script:
	"""
	python ${python_script} --path ${slidePath} \
							--xml $xml_folder \
							--level $level \
							--auto_mask ${auto_mask} \
							--size $size 
	"""
}

// Normaliser et PCAiser: à faire dans un autre script NF car cela va dépendre systematiquement du dataset à utiliser.
// Exemple : si je veux utiliser le mélance de TCGA et Curie, il faudra que je normalise sur cet ensemble.

//out .collect()
//	.into{all_wsi_tiled_1, all_wsi_tiled_2}
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
//
//	process Incremental_PCA {
//    publishDir "${output_folder}", overwrite: true, pattern: "*.txt" mode: 'copy'
//    memory '60GB'
//    cpus '16'
//
//    input:
//    val _ from all_wsi_tiled_2
//    
//    output:
//    file("*.joblib") into results_PCA
//	file("*.txt")
//
//    script:
//    output_folder = "${root_outputs}/pca/"
//	mat_folder = "${root_outputs}/mat/"
//    python_script = file("../scripts/pca_partial.py")
//    """
//    python $python_script --path ${mat_folder}
//    """
//	}
//}
