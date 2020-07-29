from tiler_wsi.tile_images_nf import ImageTiler
from tiler_wsi.arguments import get_arguments

args = get_arguments()
it = ImageTiler(args=args)
it.tile_image()