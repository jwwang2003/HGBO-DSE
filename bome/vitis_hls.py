import subprocess
from bome.log import setup_logger
import os

class VitisHLSRunner:
    def __init__(self, tcl_script=None, timeout=3600, context:str=None, log_file='runtime.log', check=False):
        self.tcl_script = tcl_script
        self.timeout = timeout
        self.check = check
        self.context = context
        self.log_file = log_file if not context else os.path.join(context, log_file)
        self.process = None
        self.stdout = ""
        self.stderr = ""
        self.error = None
        
        # Ensure context directory exists
        if self.context and not os.path.exists(self.context):
            os.makedirs(self.context)
        
        # Set up logger with detailed formatting and context-specific log file
        self.log = setup_logger(context=f"{context}")

    def run(self):
        """
        Runs the vitis_hls command with the specified TCL script or checks the version if in test mode.
        Captures stdout and stderr, and handles timeouts and errors.
        """
        try:
            if self.check:
                self.log.info("Checking if Vitis HLS is installed by checking the version...")
                self.process = subprocess.run(
                    'vitis_hls -version',
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )
                self.stdout = self.process.stdout
                self.stderr = self.process.stderr
                if self.process.returncode == 0:
                    self.log.info("Vitis HLS is installed.")
                    self.log.info(self.get_stdout())
                else:
                    self.error = "Vitis HLS version check failed."
                    self._log_error("[ERROR] Vitis HLS is not installed or version check failed!")
                    self.log.info(self.get_stdout())
                    self.log.error(self.get_stderr())
            else:
                self.process = subprocess.run(
                    f'vitis_hls -f {self.tcl_script}',
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=os.path.join("./", self.context)
                )
                self.stdout = self.process.stdout
                self.stderr = self.process.stderr
                self.log.info(self.stdout)
        except subprocess.TimeoutExpired:
            self.error = "Timeout expired"
            self._log_error("[INFO] Subprocess timeout!")
        except subprocess.SubprocessError as e:
            self.error = str(e)
            self._log_error(f"[ERROR] Subprocess error: {self.error}")
    
    def _log_error(self, message):
        """
        Logs errors to the specified log file.
        """
        self.log.error(message)
    
    def get_stdout(self):
        """
        Returns the stdout output of the vitis_hls process.
        """
        return self.stdout
    
    def get_stderr(self):
        """
        Returns the stderr output of the vitis_hls process.
        """
        return self.stderr
    
    def write_stdout_to_file(self, output_file='./vitis_hls_output.txt'):
        """
        Writes the stdout to a specified file.
        """
        with open(output_file, 'w') as file:
            file.write(self.stdout)
    
    def is_successful(self):
        """
        Returns True if the vitis_hls process ran successfully, False if there was an error or timeout.
        """
        return self.error is None

# Example usage
if __name__ == "__main__":
    # Set up logging
    # logging.basicConfig(level=logging.DEBUG)
    
    # Example of using the class with a context directory
    context_dir = './id123456'  # Example folder for storing logs
    hls_runner = VitisHLSRunner(
        tcl_script="/home/jwwang/HGBO-DSE/dse_ds/MachSuite/motpe_fl_ds/bfs/bulk/p1/script/dir_0.tcl",
        context=context_dir,
        check=True
    )
    
    # Run the vitis_hls command or check installation
    hls_runner.run()

    # Check if successful and log the output
    if hls_runner.is_successful():
        hls_runner.log.info("Vitis HLS finished successfully." if not hls_runner.check else "Vitis HLS is installed.")
        if not hls_runner.check:
            hls_runner.log.info(f"Output: {hls_runner.get_stdout()}")
            hls_runner.write_stdout_to_file()  # Optionally write to a file
    else:
        hls_runner.log.error("Vitis HLS failed.")
