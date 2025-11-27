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

import os
import time

class TimerManager:
    def __init__(self):
        self.timers = {}

    def start(self, name):
        """Start a timer with a given name."""
        self.timers[name] = time.time()

    def elapsed(self, name, reset=False):
        """Get elapsed time for the given timer."""
        if name not in self.timers:
            raise ValueError(f"Timer '{name}' not found.")
        elapsed_time = time.time() - self.timers[name]
        if reset:
            self.timers[name] = time.time()
        return elapsed_time

    def stop(self, name, print_type=None):
        """Stop the timer and optionally print the result."""
        elapsed_time = self.elapsed(name)
        if print_type is not None:
            if print_type == 'first':
                print(f"-> elapsed time: {elapsed_time:.2f} s\n")
            elif print_type == 'second':
                print(f"    -> Done in {elapsed_time:.2f} s\n")
            elif print_type == 'third':
                print(f"      -> Done in {elapsed_time:.2f} s\n")
            else:
                raise ValueError('Wrong print_type in timer.stop() argument')
        del self.timers[name]
        return elapsed_time

# Create the global timer instance
global_timer = TimerManager()


class StreamManager:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
            stream.flush()  # Important for real-time writing

    def flush(self):
        for stream in self.streams:
            stream.flush()


class SaveFolderManager:
    def __init__(self, save_folderpath, case_name):
        base = self.ensure_trailing_slash(save_folderpath)
        self.full_path = base + case_name + '/'
        self.message = None

    def ensure_trailing_slash(self, path):
        return path if path.endswith('/') else path + '/'

    def create_folder(self):
        try:
            os.makedirs(self.full_path, exist_ok=False)
            self.message = f"   Directory '{self.full_path}' created successfully.\n"
        except FileExistsError:
            self.message = f"   Directory '{self.full_path}' already exists.\n"
        except PermissionError:
            self.message = f"   Permission denied: Unable to create '{self.full_path}'.\n"
        except Exception as e:
            self.message = f"   An error occurred while creating the directory: {e}\n"