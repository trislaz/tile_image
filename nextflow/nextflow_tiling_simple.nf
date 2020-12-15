#!/usr/bin/env nextflow
glob_wsi = '/Users/trislaz/Documents/cbio/data/triples_marquages/*.svs' // insert glob pattern at the end (*.tiff for instance)
root_out = '/Users/trislaz/Documents/cbio/projets/triples_marquages/tiles'
path_mask = '/Users/trislaz/Documents/cbio/projets/triples_marquages/masks'
level = 1 
mask_level = -1
size =  2048
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
