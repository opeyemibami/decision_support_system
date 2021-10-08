FROM python:3.7-slim
#Expose port 8080
EXPOSE 8080
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./
CMD streamlit run app.py --server.port=8080 --server.address=0.0.0.0