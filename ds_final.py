from prefect import flow, task, get_run_logger
import subprocess
import os

@task(log_prints=True)  # Ensures all prints are captured in logs
def run_task(script_name):
    logger = get_run_logger()
    
    # Define the full path to the script based on the project structure
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './tasks', script_name))
    logger.info(script_path)
    try:
        # Run the external Python script using its full path
        result = subprocess.run(['python3', script_path], capture_output=True, text=True)
        
        # Log both stdout and stderr
        if result.returncode == 0:
            logger.info(f"Successfully executed {script_name}:\n{result.stdout}")  # Log standard output
        else:
            logger.error(f"Error in {script_name}: {result.stderr}")  # Log standard error

        # Always print both stdout and stderr, regardless of success or failure
        print(result.stdout)
        print(result.stderr)

    except Exception as e:
        logger.error(f"Failed to execute {script_name}: {str(e)}")

    return 0


@flow
def main_flow():
    # Run tasks sequentially and capture the results
   data1 = run_task("BasicStats.py")
   data2 = run_task("PreProcess.py", wait_for=[data1])
   data3 = run_task("EDA.py", wait_for=[data2])

# To run 
if __name__ == "__main__":
    # main_flow.serve(name="covid-ds-workflow",
    #                   tags=["covid datascience project workflow"],
    #                   parameters={},
    #                   interval=120) #2 minutes
    main_flow()