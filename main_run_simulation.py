# Copyright CETHIL UMR 5008 - INSA Lyon (2025)
#
# teddy.gresse@gmail.com
# damien.david@insa-lyon.fr
#
# IMOTEP is a micro-meteorological model that simulates the thermal environments
# of urban scenes that may contain built shelters, building overhangs or trees,
# and evaluates outdoor thermal comfort.
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software.  You can  use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

import sys
sys.path.insert(1, r'lib_imotep')
import datetime

from lib_imotep.utils import global_timer, SaveFolderManager, StreamManager
from lib_imotep.imotep import IMOTEP

def main(general_dict,
         weather_def_dict,
         panel_def_dict_list,
         airzone_def_dict_list,
         probe_set_def_list,
         output_def_dict):

    if general_dict['save_data']:
        # create output folder
        save_manager = SaveFolderManager(general_dict['save_folderpath'], general_dict['case_name'])
        save_folderpath = save_manager.full_path

        # check if the case name is different from the name of folder from which radiative viewfactors and prefactors
        # are loaded to prevent conflicts
        if general_dict['from_save']:
            if save_folderpath == general_dict['from_save_folderpath']:
                raise ValueError('The current case folder cannot be identical to the from_save_folderpath. Maybe you forgot to change the case_name.')

        save_manager.create_folder()

        # start logging
        log_path = save_folderpath + 'output.log'
        log_file = open(log_path, 'w')
        sys.stdout = StreamManager(sys.stdout, log_file)
        sys.stderr = StreamManager(sys.stderr, log_file)
    else:
        save_manager = None

    try:
        print('-----------------------------------------')
        print('RUNNING IMOTEP')
        print(f"case name: {general_dict['case_name']}")
        print(f'start datetime: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        print('-----------------------------------------\n')

        global_timer.start('global')

        # create and run IMOTEP model
        imotep = IMOTEP(
                general_dict,
                weather_def_dict,
                panel_def_dict_list,
                airzone_def_dict_list,
                probe_set_def_list,
                output_def_dict
        )

        imotep.generate()
        imotep.save_viewfactors_and_prefactors(save_manager)
        imotep.solve()
        imotep.save_case(save_manager)

        print('-----------------------------------------')
        print(f'end datetime: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'-> total elapsed time: {global_timer.stop('global'):.2f} s')
        print('-----------------------------------------')

    finally:
        # restore stdout/stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    return imotep


if __name__ == '__main__':
    ## load the inputs
    from input.urban_scene import global_input_list

    case_name_list = []
    case_name_base = None
    for i, input_tuple in enumerate(global_input_list):
        (general_dict,
         weather_def_dict,
         panel_def_dict_list,
         airzone_def_dict_list,
         probe_set_def_list,
         output_def_dict) = input_tuple

        case_name = general_dict['case_name']
        if case_name in case_name_list:
            print(f" [WARNING] case_name must be different for every simulation when doing sensitivity analysis"
                  f" -> '{'_var' + str(i + 1)}' added at the end.")
            case_name = general_dict['case_name'] + '_var' + str(i + 1)
            general_dict['case_name'] = case_name
        case_name_list.append(case_name)

        imotep = main(general_dict,
                       weather_def_dict,
                       panel_def_dict_list,
                       airzone_def_dict_list,
                       probe_set_def_list,
                       output_def_dict)