#!/usr/bin/env nextflow

levels = [1] 
size = 256 
slides_folder = "/mnt/data3/pnaylor/Data/Biopsy_Nature_3-0/*.tiff"
dataset = Channel.fromPath(slides_folder)
				 .map { file -> tuple(file.baseName, file) }
xml_folder = "/mnt/data3/pnaylor/Data/Biopsy_Nature_3-0/tissue_segmentation/"
auto_mask = 0
python_script = "/mnt/data4/tlazard/projets/snippets/tile_images/tile_images_nf.py"

process Tiling {
	publishDir "$results_folder/$slideID"
	queue 'cpu'
	stageInMode 'copy'
	memory '10GB'
	
	input:
	set slideID, file(slidePath) from dataset
	each level from levels

	output:
	set slideID, file('*.jpg') into out

	script:
	results_folder = "/mnt/data4/tlazard/data/biopsies_pet/tiled/$size/res_$level/"
	"""
	python ${python_script} --path ${slidePath} \
							--xml $xml_folder \
							--level $level \
							--auto_mask ${auto_mask} \
							--size $size 
	"""
}
