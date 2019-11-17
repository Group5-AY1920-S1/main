# CG3002 Design Project AY 19/20 Sem-1 (Group 5)

## Project Description
The wearable system detects and sends certain predefined dance moves to a server. Our system requires 5 AA batteries of at least 1.2V each to run. Another power source option could be a portable power bank with 10000mAh rated battery capacity.

## System Set Up
### A. Server 
Ensure that Python is installed. Download `final_eval_server.py` and `server_auth.py` to the same location.

### B. Arduino
1. Using the Arduino IDE, install the FreeRTOS (by Richard Barry) and Wire libraries. 
2. Download the necessary Arduino libraries provided into your Arduino’s libraries path.
3. Compile the Arduino programme `[code_arduino.ino]` and upload to the board.

### C. RPi
Add the network credentials of mobile hotspots to the ‘wpa_supplicant’ configuration file on the RPi to allow the RPi to automatically connect to the network on startup.

To remotely access the RPi, you need to use an SSH client on your laptop or PC. 

The following steps are required to SSH into the RPi:
1. Ensure that your laptop is on the same network as the RPi.
2. Open a terminal on your laptop, enter the following command , `ssh pi@[IP_ADDRESS]`, where `[IP_ADDRESS]` is the IP address of the RPi.
3. If another RPi is being used, its IP address can be found by running `hostname -i`. 
4. The password for our RPi is `rpi11`.
  
```  
Ensure that the below libraries have been installed/updated on the RPi:  
* python 3.7
* pySerial (serial communications)
* crypto 1.4.1 (authentication)
* cryptograpy 2.7 (authentication)
* pandas 0.25.1  (data processing library)
* numpy 1.16.2 (data processing library)
* scikit-learn 0.21.03 (ML library)
* scipy 1.1.0 (ML library)
```

Download the source code from the git repository into the RPi.
Once the relevant libraries have been installed, data can be collected.

## Collection of Training Data 
1. Navigate to `pi:main/Codes/comms/Rpi`.
2. Execute the command `python3 training.py`.
3. Enter the name of the dance move to train, for example `bunny`.
4. The collected data would be saved in a csv file with the dance move name entered in step 3.
5. After 1 minute has passed, the file `bunny.csv` would be generated in the Desktop directory. 

## Running the Complete System
Both the server and the RPi have to be running simultaneously.

### A. Running the server
1. Both the RPi and the server must be connected to the same network.
2. Navigate to the directory of the file "final_eval_server.py".
3. Run `python final_eval_server <IPv4 address> <port number> <group number>` to get the server running.
4. For example, if the server’s IPv4 address is `192.168.43.36`, and the port number you want to use is `6788`, and your group number is `5`, you can run `python final_eval_server 192.168.43.36 6788 5`.
   * `<IPv4 address>` can be found by running `ipconfig` on the command line. 
   * `<port number>` can be any unused port.
   * `<group number>` can be replaced by any other number as well.
5. If the RPi is already running the model, there will be a message on the server’s command line requesting a secret key used for encryption and decryption. Here, you have to enter `secretkeysixteen`. 
   * If there is no prompt, make sure that the RPi is powered on, is running the correct script, and is connected to the same network as the server.
   * Reset the Arduino and start again from step 2.
6. After the secret key is entered, get ready to dance.

### B. Running ML model (RPi)
1. Ensure the server side is ready to receive data from the RPi (from the steps above).
2. Navigate to `pi:main/Codes/comms/Rpi`.
3. Ensure that the files `client_auth.py`, `mlp.pkl` and `scaler.pkl` are also in the project directory as these files are necessary for the program to run.
4. Execute the command `python3 client_file.py`.
5. The model will now start predicting the dance moves.
6. Upon successful prediction of the logout dance move, the server will close the connection and the RPi programme will be interrupted.
   * To repeat the process, execute step 2 onwards for the server and step 4 for the RPi.
   * If the logout is sent successfully, the Arduino does not have to be physically reset; otherwise, if the programme was interrupted (for example by keyboard input), the Arduino needs to be reset.
