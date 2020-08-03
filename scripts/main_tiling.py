from tiler_wsi.tile_image.tile_images_nf import ImageTiler
from tiler_wsi.tile_image.arguments import get_arguments

args = get_arguments()
it = ImageTiler(args=args)
it.tile_image()