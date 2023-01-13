FROM python

WORKDIR C:/Users/User/Desktop/AI_Seth

COPY main.py .

RUN pip install requirements.txt

CMD ["python", "main.py"]