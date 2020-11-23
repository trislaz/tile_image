#!/usr/bin/env nextflow
glob_wsi = '/gpfsdsstore/projects/rech/gdg/uub32zv/data_sfp/train/raw/*.tif' // insert glob pattern at the end (*.tiff for instance)
root_out = '/gpfsscratch/rech/gdg/uub32zv/working_data/embedded'
path_mask = '/gpfsdsstore/projects/rech/gdg/uub32zv/data_sfp/train/masks/auto'
level = 2 
mask_level = -4
size = 256 
auto_mask = 1
tiler = 'simple' // dispo : simple | imagenet | imagenet_v2
dataset = Channel.fromPath(glob_wsi)
				 .map { file -> tuple(file.baseName, file) } 
				 .into { dataset_1; dataset_2}
root_outputs = file("${root_out}/${tiler}/size_${size}/res_${level}/")

process Tiling_folder {
	publishDir "${output_folder}", overwrite: true, pattern: "*.jpg", mode: 'copy'
	publishDir "$root_outputs/visu/", overwrite: true, pattern: "*.png", mode: 'copy'
	publishDir "$root_outputs/info/", overwrite:true, pattern: "*_infomat.npy", mode: 'copy'
	publishDir "$root_outputs/info/", overwrite:true, pattern: "*.pickle", mode: 'copy'
	publishDir "$root_outputs/info/", overwrite:true, pattern: "*.csv", mode: 'copy'

	queue "prepost"
	memory "30GB"
	maxForks 50

	input:
	set val(slideID), file(slidePath) from dataset_1

	output:
	val slideID into out
	file('*.csv')
	file('*.npy')
	file('*.pickle')
	file('*.png')
	file('*.jpg')

	script:
	slideID = slidePath.baseName
	output_folder = file("$root_outputs/${slideID}")
	python_script = file("../scripts/main_tiling.py")
	"""
	python ${python_script} --path_wsi ${slidePath} \
							--path_mask ${path_mask} \
							--level $level \
							--auto_mask ${auto_mask} \
							--tiler ${tiler} \
							--size $size \
							--mask_level ${mask_level} 
	"""
}
