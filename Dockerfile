FROM python:3.12-slim

WORKDIR / app
COPY reqirements.txt . 
RUN pip install --no-cashe-dir -r requirements.tzt 

COPY  . . 
	
CMD ["python","main.py"]

