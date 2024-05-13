"""
"""
import zipfile
import shutil
import os
import sys
import subprocess
import datetime
import argparse
import requests
from timeit import default_timer as timer
from argparse import RawTextHelpFormatter

SCRIPT_ABS_DIR = os.path.abspath(__file__).replace("reproduce_paper.py", "")

PROJECT_NAME = "DISCES"
VERSION = "0.4.1"
ZIP_LOCATION = "https://github.com/rebesatt/disces/raw/main/datasets/DISCES.zip"
FINANCE_ZIP_LOCATION = "https://github.com/rebesatt/disces/raw/main/datasets/finance/finance.zip"
GOOGLE_ZIP_LOCATION = "https://github.com/rebesatt/disces/raw/main/datasets/google/google.zip"

def print_to_std_and_file(str_to_log):
    current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    print(current_timestamp + ": " + str_to_log)
    with open(log_file, 'a') as f:
        f.write(current_timestamp + ": " + str_to_log + "\n")

def print_with_timestamp(str_to_stdout):
    current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    print(current_timestamp + ": " + str_to_stdout)

def verify_requirements():
    # TODO: Check LaTeX-Version and packages
    try:
        if not (sys.version_info.major == 3 and sys.version_info.minor >= 10):
            print_to_std_and_file("Running this script requries Python >= 3.10. Current versions is " + str(sys.version_info.major) + "." + str(sys.version_info_minor))
            sys.exit(1)
    except:
        print_to_std_and_file("Running this script requries Python >= 3.10. Please upgrade your Python and retry again.")
        sys.exit(1)
    if not shutil.which("latex"):
        print_to_std_and_file("Note: Reporducing this script requries LaTeX. Not LaTeX installation fount.")

def execute_shell_command(cmd, stdout_verification=None, b_stderr_verification=False):
    print_to_std_and_file("Executed command: " + cmd)
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout_str = str(result.stdout)[2:-1]
    stderr_str = str(result.stderr)[2:-1]
    if len(stderr_str):
        if len(stdout_str):
            print_to_std_and_file("STDOUT:")
            print_to_std_and_file(stdout_str.replace("\\r", '\r').replace("\\n", '\n'))
        print_to_std_and_file("STDERR:")
        print_to_std_and_file(stderr_str.replace("\\r", '\r').replace("\\n", '\n'))
    else:
        print_to_std_and_file(stdout_str.replace("\\r", '\r').replace("\\n", '\n'))

    if stdout_verification is not None:
        if stdout_verification not in result.stdout.decode():
            print_to_std_and_file("Verification string " + stdout_verification + " not in stdout")
            raise Exception
    if b_stderr_verification:
        if result.stderr != "b''":
            print_to_std_and_file("stderr is not empty: " + str(result.stderr[2:-1]))
            raise Exception

def execute_shell_command_with_live_output(cmd):
    print_to_std_and_file("Executed command: " + cmd)
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while True:
        output = process.stdout.readline()
        if process.poll() is not None:
            break
        if output:
            print_to_std_and_file(output.strip().decode('utf-8'))
    rc = process.poll()
    if rc != 0:
        print_to_std_and_file("RC not 0 (RC=" + str(rc) + ") for command: " + cmd)
        raise Exception
    return rc

def download_repository():
    if os.path.isdir(SCRIPT_ABS_DIR + "datasets/") and os.path.isdir(SCRIPT_ABS_DIR + "docs/") and os.path.isdir(SCRIPT_ABS_DIR + "experiments/"):
        print_to_std_and_file("The reproduce_paper.py script is part of a '" + PROJECT_NAME + "' project. No files will be downloaded.")
        project_root = SCRIPT_ABS_DIR
        #TODO:
        print_to_std_and_file("Update python-scripts for algorithms and experiments locally.")
        execute_shell_command('python src/file_updater.py')
        return project_root

    print_to_std_and_file("The reproduce_paper.py script is a standalone. Downloading '" + PROJECT_NAME + "'s source code.")
    zipped_project = SCRIPT_ABS_DIR + "/" + PROJECT_NAME + ".zip"
    project_root = SCRIPT_ABS_DIR + PROJECT_NAME + "/"

    zipped_project = 'DISCES.zip'
    print_to_std_and_file("Downloading '" + PROJECT_NAME + "'s source code ...")
    response = requests.get(ZIP_LOCATION)
    if response.status_code == 200:
        with open(zipped_project, mode='wb') as file:
            file.write(response.content)
        print_to_std_and_file("Successfully downloaded '" + PROJECT_NAME + "'s source code to " + zipped_project + ".")

        print_to_std_and_file("Extract " + zipped_project + ".")
        try:
            with zipfile.ZipFile(zipped_project, 'r') as zip_ref:
                zip_ref.extractall(".temp_" + PROJECT_NAME)
            os.remove(zipped_project)
            print_to_std_and_file("All files extracted and " + zipped_project + " deleted.")
        except Exception as e:
            print_to_std_and_file(e.message)
            sys.exit(1)
    else:
        print_to_std_and_file("Download failed. Status code " + str(response.status_code))
        sys.exit(1)

    if os.path.isdir(project_root):
        ### shutil.copytree(".temp_" + PROJECT_NAME + "/test", PROJECT_NAME, dirs_exist_ok=True)
        #TODO: remove next and uncomment second if the zip is not called 'test' anymore
        shutil.copytree(".temp_" + PROJECT_NAME + "/" + PROJECT_NAME, PROJECT_NAME, dirs_exist_ok=True)
        #shutil.copytree(".temp_" + PROJECT_NAME + "/" + PROJECT_NAME, PROJECT_NAME, dirs_exist_ok=True)
        print_to_std_and_file("Found existing local project " + project_root + ". Merge repository updates to it.")
    else:
        ### shutil.move(".temp_" + PROJECT_NAME + "/test", PROJECT_NAME)
        #TODO: remove next and uncomment second if the zip is not called 'test' anymore
        shutil.move(".temp_" + PROJECT_NAME + "/" + PROJECT_NAME, PROJECT_NAME)
        #shutil.move(".temp_" + PROJECT_NAME + "/" + PROJECT_NAME, PROJECT_NAME)
        print_to_std_and_file("Downloaded the project to " + project_root)
    shutil.rmtree(".temp_" + PROJECT_NAME)

    os.chdir(project_root)
    download_and_extract_datasets("finance", FINANCE_ZIP_LOCATION)
    download_and_extract_datasets("google", GOOGLE_ZIP_LOCATION)
    os.chdir(SCRIPT_ABS_DIR)

    return project_root

def download_and_extract_datasets(name, location):
    os.chdir("datasets/" + name + "/")
    zipped_data = name + '.zip'
    print_to_std_and_file("Downloading '" + zipped_data + "' ...")
    response = requests.get(location)
    if response.status_code == 200:
        with open(zipped_data, 'wb') as file:
            file.write(response.content)
        print_to_std_and_file("Successfully downloaded " + name + " datasets to 'datasets/" + name + "/" + zipped_data + "'.")

        print_to_std_and_file("Extract " + zipped_data + ".")
        with zipfile.ZipFile(zipped_data, 'r') as zip_ref:
            zip_ref.extractall(".temp_" + name)
        shutil.copytree(".temp_" + name + "/", "../" + name, dirs_exist_ok=True)
        os.remove(zipped_data)
        shutil.rmtree(".temp_" + name)
        print_to_std_and_file("All files extracted and " + zipped_data + " deleted.")
    else:
        print_to_std_and_file("Download of " + zipped_data + " failed. Status code " + str(response.status_code))

    os.chdir("../../")

def create_synthetic_data():
    cur_wd = os.getcwd()
    os.chdir("datasets/synt/")
    execute_shell_command_with_live_output("python synt_sample_generator.py")
    os.chdir(cur_wd)

def create_verification_file(local_result_dir, file_to_verify_execution):
    f = open(local_result_dir + "/" + file_to_verify_execution, "w")
    f.write("Done")
    f.close()

def run_experiments(local_result_dir, run_command_prefix, functions, test_name_prefix, file_to_verify_execution, args=None, estimated_runtimes=None):
    if os.path.isfile(local_result_dir + "/" + file_to_verify_execution):
        print_to_std_and_file("Found existing test folder. Skipping.")
    else:
        b_create_verification_file = True
        for i, function in enumerate(functions):
            cmd = str(run_command_prefix) + " " + str(test_name_prefix) + str(function) + ".py"
            if args:
                cmd += " " + str(args[i])
            if estimated_runtimes:
                print_to_std_and_file("Experiment " + str(test_name_prefix) + str(function) + " estimated runtime is " + str(estimated_runtimes[i]) + ".")
            try:
                start = timer()
                execute_shell_command_with_live_output(cmd)
                end = timer()
                print_to_std_and_file("Experiment " + str(test_name_prefix) + str(function) + " took " + str(end - start) + " seconds")
            except Exception as msg:
                end = timer()
                print_to_std_and_file(str(msg))
                print_to_std_and_file("Experiment " + str(test_name_prefix) + str(function) + " failed after " + str(end - start) + " seconds")
                b_create_verification_file = False
        if b_create_verification_file:
            create_verification_file(local_result_dir, file_to_verify_execution)
    assert os.path.isfile(local_result_dir + "/" + file_to_verify_execution)

def compile_main_pdf():
    figure_list = {"Figure 1": "sota_4_broken.pdf",
                   "Figure 2": "rl_compare.pdf",
                   "Figure 3": "synt_plots.pdf",
                   "Figure 4": "cluster.pdf",
                   "Figure 5": "exclude.pdf",
                   "Figure 6": "sota_acc.pdf"}
    if not os.path.isdir(project_root + "docs/latex_src/img"):
        print_to_std_and_file("'" + project_root + "docs/latex_src/img' was not available. All original images are missing.")
        os.makedirs(project_root + "docs/latex_src/img")
        print_to_std_and_file("'" + project_root + "docs/latex_src/img' created.")
    for figure_name, figure_file in figure_list.items():
        new_file_exists = os.path.isfile(local_result_dir + "/" + figure_file)
        cur_file_exists = os.path.isfile(project_root + "docs/latex_src/img/" + figure_file)
        old_file_exists = os.path.isfile(project_root + "docs/latex_src/original_img/" + figure_file)
        if new_file_exists:
            if cur_file_exists and not old_file_exists:
                if not os.path.isdir(project_root + "docs/latex_src/img"):
                    os.makedirs(project_root + "docs/latex_src/original_img")
                    print_to_std_and_file("'" + project_root + "docs/latex_src/original_img' created.")
                shutil.copyfile(project_root + "docs/latex_src/img/" + figure_file, project_root + "docs/latex_src/original_img/" + figure_file)
                print_to_std_and_file("Replacing " + figure_name + " (" + figure_file + ") with the reproduced one. Original file can be found in '" + project_root + "docs/latex_src/original_img/")
            else:
                print_to_std_and_file("Copying " + figure_name + " (" + figure_file + ") to '" + project_root + "docs/latex_src/img/")
                print_to_std_and_file("Note: original " + figure_name + " (" + figure_file + ") is not available.")
            shutil.copyfile(local_result_dir + "/" + figure_file, project_root + "docs/latex_src/img/" + figure_file)
        else:
            if cur_file_exists and not old_file_exists:
                print_to_std_and_file("Note: " + figure_name + " (" + figure_file + ") wasn't found in " + local_result_dir + ". Using the original figure from the paper.")
            if not cur_file_exists:
                if old_file_exists:
                    print_to_std_and_file("Copying " + figure_name + " (" + figure_file + ") to '" + project_root + "docs/latex_src/img/")
                    print_to_std_and_file("Note: " + figure_name + " (" + figure_file + ") wasn't found in " + local_result_dir + ". Using the original figure from the paper.")
                    shutil.copyfile(project_root + "docs/latex_src/original_img/" + figure_file, project_root + "docs/latex_src/img/" + figure_file)
                else:
                    print_to_std_and_file("Note: " + figure_name + " (" + figure_file + ") wasn't found in '" + project_root + "docs/latex_src/original_img/'." +
                    " Plot will not be included in the paper.")

    os.chdir(project_root + "docs/latex_src/")

    execute_shell_command('pdflatex --interaction=nonstopmode main.tex')
    execute_shell_command('bibtex main.aux')
    execute_shell_command('pdflatex --interaction=nonstopmode main.tex')
    execute_shell_command('pdflatex --interaction=nonstopmode main.tex', stdout_verification="main.pdf")

    shutil.copyfile("main.pdf", reproduced_main_pfd_file)
    print_to_std_and_file("The reproduced paper, containing the new figures based on these experiments, is in " + reproduced_main_pfd_file)

def clean_working_directory():
    try:
        os.remove("reproduce_paper.log")
    except:
        pass
    shutil.rmtree("DISCES/", ignore_errors=True)
    shutil.rmtree("datasets/synt/domain_size/", ignore_errors=True)
    shutil.rmtree("datasets/synt/max_pattern/", ignore_errors=True)
    shutil.rmtree("datasets/synt/max_type/", ignore_errors=True)
    shutil.rmtree("datasets/synt/trace_length/", ignore_errors=True)
    shutil.rmtree("datasets/synt/traces/", ignore_errors=True)
    shutil.rmtree("datasets/synt/types/", ignore_errors=True)
    #TODO:
    #docs latexmk -C
    #execute_shell_command('pdflatex --interaction=nonstopmode main.tex', stdout_verification="main.pdf")
    shutil.rmtree("experiments/experiment_results/", ignore_errors=True)

if __name__ == "__main__":
    # Setup
    parser = argparse.ArgumentParser(description =  "Reproduce ...\n"
            "Requirements:\n"
            "(1) Python >= 3.10",
            formatter_class=RawTextHelpFormatter)
    parser.add_argument("--dd", dest="b_download_dataset",
            help="if --dd is specified, the script only downloads the repository and external datasets and exits (without running any experiments)",
            action='store_true')
    parser.add_argument("--ilm", dest="b_run_extended_il_miner",
            help="",
            action="store_true")
    parser.add_argument("--noexp", dest="b_run_no_experiment",
            help="",
            action="store_true")
    parser.add_argument("--C", dest="b_clean_working_directory",
            help="",
            action="store_true")
    #TODO: parser.add_argument("--vv", dest="b_run_extended_il-miner", help="", action="store_true")
    args = parser.parse_args()

    if args.b_clean_working_directory:
        clean_working_directory()
        sys.exit()

    log_file = os.path.abspath(__file__).replace("reproduce_paper.py", "reproduce_paper.log")
    reproduced_main_pfd_file = os.path.abspath(__file__).replace("reproduce_paper.py", "main.pdf")
    print_to_std_and_file("======================== Reproduce '" + PROJECT_NAME + "'s Experiments ========================")
    print_with_timestamp("The script log is at " + str(log_file))

    verify_requirements()

    project_root = download_repository()
    os.chdir(project_root)

    create_synthetic_data()
    if args.b_download_dataset:
        sys.exit()

    # Run the experiments
    try:
        local_result_dir = os.getcwd() + "/experiments/experiment_results"
        try:
            os.makedirs(local_result_dir)
            print_to_std_and_file("'" + local_result_dir + "' created.")
        except FileExistsError:
            pass
        if not args.b_run_no_experiment:
            functions = {# "[Name]" : "[Runtime]"
                    "cluster" : "85.200 seconds (23:40:00 h)",
                    "correlation" : "2.400 seconds (00:40:00 h)",
                    "exclude" : "8.400 seconds (02:20:00 h)",
                    "rl_compare" : "-",
                    "sota_4_broken" : "1.320 seconds (00:22:00 h)",
                    "sota_acc" : "740 seconds (00:12:20 h)",
                    "synt_plot" : "2.700 seconds (00:45:00 h)",
                    }
            run_ilm = ""
            if args.b_run_extended_il_miner:
                run_ilm = "--ilm"
            args = ["", "", "", "", run_ilm, "", ""]
            assert len(functions) == len(args)
            try:
                run_experiments(local_result_dir, "python", functions.keys(), "experiments/", ".verification_file", args=args, estimated_runtimes=list(functions.values()))
            except AssertionError:
                print_to_std_and_file("Experiments did not finish due to an error! Check the log for more details.")

        # Compile .tex to .pdf
        print_to_std_and_file("Compiling the paper.pdf...")
        compile_main_pdf()
    finally:
        os.chdir(project_root)
