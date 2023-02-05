import os
import sys

from PIL import Image
from tqdm import tqdm

from modules import shared, images, devices, scripts, scripts_postprocessing, ui_common, generation_parameters_copypaste
from modules.shared import opts


def run_postprocessing(extras_mode, image, image_folder, input_dir, output_dir, show_extras_results, recurse_subdirs, preserve_file_names, skip_existing, *args, save_output: bool = True):
    devices.torch_gc()

    shared.state.begin()
    shared.state.job = 'extras'

    image_data = []
    image_names = []
    outputs = []

    if extras_mode == 1:
        for img in image_folder:
            image = Image.open(img)
            image_data.append(image)
            image_names.append(os.path.splitext(img.orig_name)[0])
    elif extras_mode == 2:
        assert not shared.cmd_opts.hide_ui_dir_config, '--hide-ui-dir-config option must be disabled'
        assert input_dir, 'input directory not selected'

        input_dir = os.path.abspath(input_dir)
        len_input_dir = len(input_dir)
        image_list = shared.listfiles(input_dir, recursive=recurse_subdirs)
        for filename in image_list:
            try:
                image = Image.open(filename)
            except Exception:
                continue
            image_data.append(image)
            image_names.append(filename)
    else:
        assert image, 'image not selected'

        image_data.append(image)
        image_names.append(None)

    if extras_mode == 2 and output_dir != '':
        outpath = output_dir
    else:
        outpath = opts.outdir_samples or opts.outdir_extras_samples

    infotext = ''

    for image, name in zip(image_data, tqdm(image_names)):
        shared.state.textinfo = name

        existing_pnginfo = image.info or {}

        pp = scripts_postprocessing.PostprocessedImage(image.convert("RGB"))

        if save_output:
            forced_filename = None
            final_outpath = os.path.abspath(outpath)

            if recurse_subdirs:
                orig_file_dir = os.path.split(name)[0]
                final_outpath = os.path.join(outpath, orig_file_dir[len_input_dir + 1:])

                if preserve_file_names and name is not None:
                    forced_filename, ext = os.path.splitext(os.path.basename(name))

                if forced_filename is not None and final_outpath == orig_file_dir:
                    raise Exception('out dir is the same as in dir and preserve names is checked, original files could be overwritten, aborting.')

                forced_filename_with_ext = f'{forced_filename}{ext}'
                if forced_filename is not None and skip_existing and os.path.exists(os.path.join(final_outpath, forced_filename_with_ext)):
                    print(f'destination file "{forced_filename_with_ext}" exists, skipping')
                    continue

        try:
            scripts.scripts_postproc.run(pp, args)
        except:
            if not recurse_subdirs:
                raise
            print(sys.exc_info())
            print(f'error ocurred on file "{name}", but recursive, so continuing')
            continue

        if opts.use_original_name_batch and name is not None:
            basename = os.path.splitext(os.path.basename(name))[0]
        else:
            basename = ''

        infotext = ", ".join([k if k == v else f'{k}: {generation_parameters_copypaste.quote(v)}' for k, v in pp.info.items() if v is not None])

        if opts.enable_pnginfo:
            pp.image.info = existing_pnginfo
            pp.image.info["postprocessing"] = infotext

        if image.mode == 'RGBA':
            try:
                img_alpha = Image.new('RGB', image.size)
                img_alpha.putdata(image.getchannel('A').getdata())
                pp_alpha = scripts_postprocessing.PostprocessedImage(img_alpha)
                scripts.scripts_postproc.run(pp_alpha, args)
                pp.image.putalpha(pp_alpha.image.getchannel('R'))
            except:
                print(f'error adding alpha for "{name}"')

        if save_output:
            images.save_image(pp.image, path=final_outpath, basename=basename, seed=None, prompt=None, extension=opts.samples_format, info=infotext, short_filename=True, no_prompt=True, grid=False, pnginfo_section_name="extras", existing_info=existing_pnginfo, forced_filename=forced_filename)

        if extras_mode != 2 or show_extras_results:
            outputs.append(pp.image)

    devices.torch_gc()

    return outputs, ui_common.plaintext_to_html(infotext), ''


def run_extras(extras_mode, resize_mode, image, image_folder, input_dir, output_dir, show_extras_results, gfpgan_visibility, codeformer_visibility, codeformer_weight, upscaling_resize, upscaling_resize_w, upscaling_resize_h, upscaling_crop, extras_upscaler_1, extras_upscaler_2, extras_upscaler_2_visibility, upscale_first: bool, save_output: bool = True):
    """old handler for API"""

    args = scripts.scripts_postproc.create_args_for_run({
        "Upscale": {
            "upscale_mode": resize_mode,
            "upscale_by": upscaling_resize,
            "upscale_to_width": upscaling_resize_w,
            "upscale_to_height": upscaling_resize_h,
            "upscale_crop": upscaling_crop,
            "upscaler_1_name": extras_upscaler_1,
            "upscaler_2_name": extras_upscaler_2,
            "upscaler_2_visibility": extras_upscaler_2_visibility,
        },
        "GFPGAN": {
            "gfpgan_visibility": gfpgan_visibility,
        },
        "CodeFormer": {
            "codeformer_visibility": codeformer_visibility,
            "codeformer_weight": codeformer_weight,
        },
    })

    return run_postprocessing(extras_mode, image, image_folder, input_dir, output_dir, show_extras_results, *args, save_output=save_output)
