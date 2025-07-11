Task: get dev/yes-no-maybe-decoupled.py working.

Step 1: Familiarize yourself with the codebase.

Step 2: Familiarize yourself with the new decoupled Unsloth service in /src/art/unsloth/decoupled-service.py, as well as with the other files in /src/art/unsloth/

Step 3: In a loop do the following:

- Run the script dev/yes-no-maybe-decoupled.py in the background saving the output to a file because it could be a long running process.
- Sleep for 30-60 seconds and read the process output.
- If everything is working, sleep again and repeat the process.
- If the process has crashed or is hanging with some sort of error, kill the process (scripts/kill-gpu-processes.sh may be handy) and fix the error.
- Repeat until the model successfully goes through several iterations and the reward increases meaningfully.