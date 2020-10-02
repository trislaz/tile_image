#!/usr/bin/env nextflow

glob_wsi = '/mnt/data4/gbataillon/Dataset/Curie/Global/raw/*.ndpi' // insert glob pattern at the end (*.tiff for instance)
root_out = '/mnt/data4/tlazard/data/curie/curie_recolo_tiled'
path_mask = '/mnt/data4/tlazard/data/curie/curie_recolo_annot/auto'
level_list = 0 
mask_level = -1
size = 256 
auto_mask = 1
tiler = 'simple' // dispo : simple | imagenet | imagenet_v2
dataset = Channel.fromPath(glob_wsi)
				 .map { file -> tuple(file.baseName, file) } 
				 .into { dataset_1; dataset_2}
root_outputs = file("${root_out}/${tiler}/size_${size}/res_${level}/")

process Tiling_folder {
	publishDir "${output_folder}", overwrite: true, pattern: "*.jpg", mode: 'copy'
	publishDir "${output_folder}", overwrite: true, pattern: "tile_*.npy", mode: 'copy'
	publishDir "$root_outputs/visu/", overwrite: true, pattern: "*.png", mode: 'copy'
	publishDir "$root_outputs/info/", overwrite:true, pattern: "*_infomat.npy", mode: 'copy'
	publishDir "$root_outputs/info/", overwrite:true, pattern: "*.pickle", mode: 'copy'
	publishDir "$root_outputs/info/", overwrite:true, pattern: "*.csv", mode: 'copy'

	queue "gpu-cbio"
	maxForks 10
    clusterOptions "--gres=gpu:1"
    memory '20GB'

	input:
	set val(slideID), file(slidePath) from dataset_1

	output:
	val slideID into out
	file('*.csv')
	file('*.npy')
	file('*.pickle')
	file('*.png')
	if (tiler == 'simple'){
		file('*.jpg')
	}

	script:
	slideID = slidePath.baseName
	output_folder = file("$root_outputs/${slideID}")
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
