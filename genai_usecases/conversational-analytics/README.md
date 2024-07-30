# Conversational Analytics

![Conversational Analytics](https://via.placeholder.com/600x200?text=Conversational+Analytics)

Welcome to the **Conversational Analytics** project. This sample application lets customers send feedback while our system identifies sentiment, topics, and emerging trends using a Large Language Model (LLM). The project is still in development, but the basic features are ready for you to use. Currently, it combines analytics for all users, but in the next update, I will separate the analytics for each user and add more important features related to conversational analytics. Feel free to use this repo for your experiments and customize it to meet your needs.

## 🚀 Project Details

This project enables conversational analytics by analyzing customer feedback to uncover sentiment and trends. It leverages the power of Google's free Gemini-pro LLM API, providing valuable insights to help you understand your customers better.

### 🔥 Key Features

- Real-time sentiment analysis
- Topic detection and categorization
- Trend identification and tracking
- Role-based access control (Admin and User roles)
- More...Work in Progress .........

## 🛠️ Technology | Tool Stack

- **Frontend**: React
- **Backend**: Python, FastAPI, uvicorn
- **Database**: Mongodb (Uses inside docker so don't need to install separate at your system. Just use deafult URL given at the project)
- **AI Model**: Google's Gemini-pro LLM API
- **Containerization**: Docker (Need to install at your laptop/desktop)
- **Authentication**: Custom JWT implementation
- **Postman**: To create Admin user using rest api call with X-Admin-Key (SECRET_KEY). You can use CURL command as well if you have access to CURL at your system
## 📂 Project Structure
```
Conversational-analytics/
├── backend/
│ ├── auth.py
│ ├── main.py
│ └── Dockerfile
├── frontend/
│ ├── public/
│ └── src/
│ ├── components/
│ │ ├── AnalyticsDisplay.tsx
│ │ ├── FeedbackDisplay.tsx
│ │ ├── FeedbackForm.tsx
│ │ ├── Home.tsx
│ │ ├── Login.tsx
│ │ ├── Footer.tsx
│ │ ├── Header.tsx
│ │ ├── PrivateRoute.tsx
│ │ └── Register.tsx
│ ├── App.css
│ ├── App.tsx
│ ├── index.css
│ ├── index.tsx
│ └── Dockerfile
├── .env
├── docker-compose.yml
├── secret_key_generation.py
└── Conversational-Analytics.postman_collection.json
```

## 🚀 Getting Started

### Prerequisites

- Git
- Docker
- Postman (for API testing)
- Google Cloud account (for Gemini-pro API key)

### Installation

1. Clone the repository and open the foder at your IDE or terminal:

```bash
git clone https://github.com/yourusername/Conversational-analytics.git
cd Conversational-analytics
```
2. Set up your Google Gemini-pro API key:
- Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
- Create a new API key
- Add the key to your `.env` file: `GEMINI_API_KEY=your_api_key_here`
- How to setup Google's free Gemini Pro API Key - [![YouTube](https://img.shields.io/badge/YouTube-Video-green)](https://www.youtube.com/watch?v=ZHX7zxvDfoc)

3. Generate a secret key for creating Admin User in other step:

```python
import secrets

secret_key = secrets.token_hex(32)  # Generates a 64 character hex string

print(secret_key)
```
Copy the output and add it to your `.env` file: `SECRET_KEY=generated_secret_key`

4. Build and run the Docker containers at your project folder:
Example:
```bash
c:/projects/conversational-analytics> docker-compose up --build
```
To close the application. Type **Ctrl+C** to stop the session (Simplest way)

To remove the application specific docker components, run following command
Example:
```bash
c:/projects/conversational-analytics> docker-compose down
```
5. Create an admin user using Postman:
- Import the `Conversational-Analytics.postman_collection.json` file
- Update the `X-Admin-Key` header with your `SECRET_KEY`
- Send the request to create an admin user

6. Access the application:
- Frontend: `http://localhost:3000/login`
- Backend API: `http://localhost:8000`

## 👥 User Roles

1. **Admin**: Full access to analytics and feedback management
2. **User**: Can submit feedback through the user interface

## 🔄 Development Workflow

1. Make changes to the codebase
2. Rebuild and restart containers: `docker-compose up --build`
3. Test your changes
4. Commit and push to your repository

## 📸 Screenshots

1. Login <br>
<img width="400" src="https://github.com/genieincodebottle/generative-ai/blob/main/genai_usecases/conversational-analytics/images/login.png">

2. Register<br>
<img width="400" src="https://github.com/genieincodebottle/generative-ai/blob/main/genai_usecases/conversational-analytics/images/register.png">

3. User Home Page<br>
<img width="400" src="https://github.com/genieincodebottle/generative-ai/blob/main/genai_usecases/conversational-analytics/images/user-home.png">

4. Admin Home Page<br>
<img width="400" src="https://github.com/genieincodebottle/generative-ai/blob/main/genai_usecases/conversational-analytics/images/admin-home.png">

5. User Feedback<br>
<img width="400" src="https://github.com/genieincodebottle/generative-ai/blob/main/genai_usecases/conversational-analytics/images/user-feedback.png">

6. Analytics<br>
<img width="400" src="https://github.com/genieincodebottle/generative-ai/blob/main/genai_usecases/conversational-analytics/images/analytics.png">

7. Feedback Details<br>
<img width="400" src="https://github.com/genieincodebottle/generative-ai/blob/main/genai_usecases/conversational-analytics/images/feedback-details.png">

## 🛣️ Roadmap

- [ ] Multi-user support with segregated feedback
- [ ] Enhanced analytics capabilities
- [ ] Integration with popular CRM systems
- [ ] Mobile app development
- [ ] AI-powered chatbot for automated feedback collection

---

Let me know if you face any issue running this application at your system.

Happy Coding! 🎉