import copernicusmarine as cm

output_directory = "data/wave_height/"
output_filename = "cmems_mod_glo_wav_my_0.2deg_PT3H-i_VHM0_15.00W-15.00E_35.00N-65.00N_2010-01-01-2021-01-01.nc"

cm.subset(
    dataset_id="cmems_mod_glo_wav_my_0.2deg_PT3H-i",
    variables=["VHM0"],
    minimum_longitude=-15,
    maximum_longitude=15,
    minimum_latitude=35,
    maximum_latitude=65,
    start_datetime="2010-01-01T12:00:00",
    end_datetime="2021-01-01T12:00:00",
    output_directory=output_directory,
    output_filename=output_filename,
)
