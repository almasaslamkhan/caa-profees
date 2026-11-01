# 🚀 Profees Model API & Mock API Deployment

This folder is responsible for deploying the **Profees Model API** and **Mock API**.  
Every time a commit is made as specific branch, **ocibuild.conf** allows building and publishing the Docker image defined in:  📂 `cdrc/profees_coding/deploy/api/Dockerfile`

---

## 📌 API Development & Deployment Steps

### **1️⃣ Run the Dockerfile**
For **production integration**, the engineering team should follow these steps:

#### **Get the Dockerfile**
- Place the `Dockerfile` in the **PA model service repository**.
- Use one of the **two available CDRC images**.

#### **Dockerfile Example**
```dockerfile
# Use the CDRC image from two optional Load Balancers (LB)
FROM odo-docker-signed-local.artifactory-builds.oci.oraclecorp.com/clinical-rev-cycle-profees-api:0.0.32
# Alternative option:
# FROM odo-docker-signed-local.artifactory.oci.oraclecorp.com/clinical-rev-cycle-profees-api:0.0.32

```

📌 Note: The image version specified at **ocibuild.conf** can update frequently—please confirm the version with the Science team before deployment.

### **2️⃣ Build the Docker Image**

Run the following command to build the image:
```bash
docker build -t profees -f cdrc/profees_coding/deploy/api/Dockerfile .
```
📌 Note: The cdrc/profees_coding/deploy/api/Dockerfile is the specified Dockerfile from the previous step.


### **3️⃣ Run the Docker Container**

Run the Model API
```bash
docker run -d -e MODE_PROFEES=PROD -p 8002:8002 --name prodapi profees
```
📌 Note: when MODE_PROFEES=MOCK is not specified, MOCK response would be provided.

### **4️⃣ Pull & Run the Docker Image Directly**

Instead of building the image manually, you can pull and run it directly:

Pull the latest image

```bash
docker pull odo-docker-signed-local.artifactory-builds.oci.oraclecorp.com/clinical-rev-cycle-profees-api:0.0.32
```

🛠 Additional Notes
	•	PROD Mode (MODE_PROFEES=PROD) → Runs the production API instance.
	•	DEV Mode (MODE_PROFEES=DEV) → Runs the API for development.
	•	TEST_PROFEES Mode (MODE_PROFEES=TEST_PROFEES) → Runs an API in LOCAL or SDS environment for Science team testing.
	•	ENV_PROFEES=SDS → Specifies the environment setting.

📌 Ensure the correct environment variables are set before running the containers.

### **5️⃣ Test the APIs**
After starting the API containers, you can test their functionality by invoking them with Python.

```bash
python cdrc/profees_coding/deploy/api/invoke/invoke.py --port 8002
```
📌 Note: Ensure the API containers are running before testing.