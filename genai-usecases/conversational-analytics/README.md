<img src="../../genai-usecases//conversational-analytics/images/conversational-analytics.png">

Welcome to the fullstack Conversational Analytics Generative AI application. This app allows customers to share feedback while the system automatically detects Key Topics, Overall Sentiment, and Emerging Trends using a Large Language Model (LLM).

Feel free to use this repo for your own experiments and customize it to fit your needs.
## 🎥 YouTube
[![YouTube Video](https://img.youtube.com/vi/fzkM-qkibpM/0.jpg)](https://www.youtube.com/watch?v=fzkM-qkibpM)

## Project Details

This project enables conversational analytics by analyzing customer feedback to uncover important topics, sentiment and emerging-trends. It leverages the power of Google's free Gemini-pro LLM API, providing valuable insights to help you understand your customers better. You can replace gemini-pro LLM API endpoint with any other LLM at main.py file.

**Example:**

**Customer Feedback:** I reached out for help with my account, and the support team was very responsive and helpful. I appreciate their quick assistance, but it would be helpful to have more self-service options available.

**Conversational Analytics:**

    Key Topics: Account support

    Overall Sentiments: Mixed (Positive | Negative)

    Emerging Trends: Limited self-service options
---

### 🔥 Key Features

- Real-time sentiment analysis
- Topic detection and categorization
- Trend identification and tracking
- Role-based access control (Admin and User roles)

---
### 🛠️ Technology | Tool Stack

- **Frontend**: React
- **Backend**: Python, FastAPI, uvicorn
- **Database**: Mongodb (Uses inside docker so don't need to install separate at your system. Just use default URL given at the project)
- **AI Model**: Google's Gemini-pro LLM API
- **Containerization**: Docker (Need to install at your laptop/desktop)
- **Authentication**: Custom JWT implementation
- **Postman**: To create Admin user using rest api call with X-Admin-Key (SECRET_KEY). You can use CURL command as well if you have access to CURL at your system

---
### 📂 Project Structure
```
Conversational-analytics/
├── backend/
│ ├── test
│ ├── .gitignore
│ ├── auth.py
│ ├── Dockerfile
│ ├── main.py
│ └── requirements.txt
├── frontend/
│ ├── public/
│ └── src/
│ │ ├── components/
│ │ │ ├── AnalyticsDisplay.tsx
│ │ │ ├── FeedbackDisplay.tsx
│ │ │ ├── FeedbackForm.tsx
│ │ │ ├── Home.tsx
│ │ │ ├── Login.tsx
│ │ │ ├── Footer.tsx
│ │ │ ├── Header.tsx
│ │ │ ├── PrivateRoute.tsx
│ │ │ └── Register.tsx
│ │ ├── App.css
│ │ ├── App.tsx
│ │ ├── index.css
│ │ ├── index.tsx
│ │ ├── logo.svg
│ │ ├── react-app-env.d.ts
│ │ ├── reportWebVitals.ts
│ │ └── types.ts
│ ├── .gitignore
│ ├── Dockerfile
│ ├── package-lock.json
│ ├── package.json
│ ├── README.md
│ └── tsconfig.json
├── .env
├── .gitignore
├── Conversational-Analytics.postman_collection.json
├── docker-compose.yml
├── README.md
└── secret_key_generation.py
```
---
### Getting Started

#### Prerequisites

- [Git](https://git-scm.com/downloads)
- [Docker Desktop](https://docs.docker.com/engine/install/)
- [Postman (for API testing)](https://www.postman.com/downloads/)
- [Free Google's Gemini API Key](https://makersuite.google.com/app/apikey)

#### Installation

1. Clone the repository and open the folder at your IDE or terminal:

    ```bash
    git clone https://github.com/genieincodebottle/generative-ai.git
    cd generative-ai\genai-usecases\conversational-analytics
    ```
2. Configure Environment
      * Rename .env.example → .env
      * Update with your keys:

        ```bash
        GOOGLE_API_KEY=your_key_here # Using the free-tier API Key
        ```
        * Get **GOOGLE_API_KEY** here -> https://aistudio.google.com/app/apikey

3. Generate a secret key to create Admin User in other step:

    ```python
    import secrets

    secret_key = secrets.token_hex(32)  # Generates a 64 character hex string

    print(secret_key)
    ```
    Copy the print output and add it to your `.env` file at following key: 

    `SECRET_KEY=<generated_secret_key>`
    

4. Build and run the Docker containers at your project folder:
    
    ```bash
    c:/<your_folder_location>/conversational-analytics> docker-compose up --build
    ```
    > To close the application. Type **Ctrl+C** to stop the session or from Docker Desktop UI

    > To remove the application from docker, run following command or use Docker Desktop UI
     
        c:/<your_folder_location>/conversational-analytics> docker-compose down
        
5. Create an admin user using Postman (You can create Normal User using Register link at UI but to create Admin user, you need to run Rest API call using postman or using CURL command):
    - Import the `Conversational-Analytics.postman_collection.json` file at your local postman
    - After import, update the `X-Admin-Key` header with your `SECRET_KEY`
    - Send the request to create an admin user
    - Screenshots
    <br>
    <img width="400" src="../../genai-usecases/conversational-analytics/images/postman-1.png">
    <br>
    <img width="400" src="../../genai-usecases/conversational-analytics/images/postman-2.png">

    Note: If you want to use curl command instead of postman then run following curl command after changing X-Admin-key value and username & password
    ```bash
    curl --location 'http://localhost:8000/register' \
    --header 'Content-Type: application/json' \
    --header 'X-Admin-Key: <your secret-key generated in earlier step using python code>' \
    --data '{
        "username": "suresh",
        "password": "suresh"
    }'
    ```
    
6. Access the application:
    - Frontend: `http://localhost:3000/login`
    - Backend API: `http://localhost:8000`

---
### 👥 User Roles

1. **Admin**: Full access to analytics and feedback management
2. **User**: Can submit feedback through the user interface

---
### 🔄 Development Workflow

1. Make changes to the codebase as per your requirement like adding SECRET_KEY, GOOGLE_API_KEY
2. Rebuild and restart containers: `docker-compose up --build`
3. Test your changes

---
### 📸 Screenshots

1. Login <br>
<img width="400" src="../../genai-usecases/conversational-analytics/images/login.png">

2. Register<br>
<img width="400" src="../../genai-usecases/conversational-analytics/images/register.png">

3. User Home Page<br>
<img width="400" src="../../genai-usecases/conversational-analytics/images/user-home.png">

4. Admin Home Page<br>
<img width="400" src="../../genai-usecases/conversational-analytics/images/admin-home.png">

5. User Feedback<br>
<img width="400" src="../../genai-usecases/conversational-analytics/images/user-feedback.png">

6. Analytics<br>
<img width="400" src="../../genai-usecases/conversational-analytics/images/analytics.png">

7. Feedback Details<br>
<img width="400" src="../../genai-usecases/conversational-analytics/images/feedback-details.png">

---

Let me know if you face any issue running this application at your system.

Happy Coding! 🎉