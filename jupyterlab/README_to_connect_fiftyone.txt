# This is to remotely connect fiftyone app and notebook from local machine/client.

# To start Jupyter server on remote machine
jupyter-lab --ip 0.0.0.0 --port 8887--no-browser ./fo.ipynb #It's the /path/to/notebook.ipynb

# On local machine/client connect to Jupyter server above
ssh -N -L 8887:127.0.0.1:8887 fredrik@192.168.1.44

# On local machine/client to forward from notebook fo.launch_app port 5151
# to local machine browser tab 5151 fiftyone app 
ssh -N -L 5151:127.0.0.1:5151 fredrik@192.168.1.44

# Connect to jupyter server from local machine/client
http://alix-ubuntu20:8887/lab?token=b1d8e8142d63eb706b1124eab74cc223a31f378dc1f69878

