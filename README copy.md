
DOCKER gpu로 돌리기 (Window desktop app일 경우)

<Method 1>
Use the Docker Desktop Dashboard
Open Docker Desktop:

(1) Left-click on the Docker icon in the system tray. This action should open the Docker Desktop dashboard.
Alternatively, you can search for "Docker Desktop" in the start menu and open it from there.
Access Settings:

(2) In the Docker Dashboard, look for a gear icon or a menu item labeled "Settings" or "Preferences". This is usually found in the top right corner of the Docker Dashboard window.
Method 2: Double-Checking the System Tray
Ensure that you are clicking on the correct icon:

(3) The Docker icon typically appears as a whale with containers on its back. Sometimes, multiple icons may be similar, or the Docker icon may change slightly with updates.
Try hovering your mouse over icons to see tooltip texts that might identify them as Docker.
Method 3: Restart Docker
If the Docker icon isn't behaving as expected:

(4) Right-click the Docker icon and select "Restart Docker" or similar options to restart. After a restart, check again if the settings option appears.
Method 4: Check for Updates
Ensure your Docker Desktop is up to date, as interfaces can change with updates:

(5) Open Docker Desktop using any method you can (through the start menu, for instance).
Look for any notifications or options within the application suggesting updates are available and update if necessary.
Method 5: Reinstall Docker
If all else fails:

(6) Consider uninstalling and then reinstalling Docker Desktop. Sometimes installation errors or updates can cause issues with how applications appear and behave in the system tray.
By trying these methods, you should be able to access Docker settings and configure it according to your needs. If issues persist, checking online for specific Docker Desktop version issues or contacting Docker's support may also be beneficial.

<도커 gpu로 돌릴떄 명령어>
wsl - 윈도우창으로 리눅스를 건드릴수 있는 명령창을 띄워준다

# Stop and remove the existing container if needed
docker stop ollama
docker rm ollama

# Run the container with GPU support
docker run --gpus all -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# Execute the command inside the container (올라마 이미지 안에서 mistral run 시키면 된다)
docker exec -it ollama ollama run mistral


귀찮으면 스크립트 만들어서 사용 ㄱ