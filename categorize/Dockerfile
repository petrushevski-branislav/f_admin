FROM python:3.8-slim

# copy local python scripts into container
WORKDIR /src
ADD consumer.py get_ip.py requirements.txt /src/

# create virtualenv
RUN pip3 install -r requirements.txt \
    && pip3 list

# puts virtualenv utils at top of path
# commands from inside container will run venv versions
# https://stackoverflow.com/questions/44077407/is-there-a-way-to-automatically-activate-a-virtualenv-as-a-docker-entrypoint
# https://pythonspeed.com/articles/activate-virtualenv-dockerfile/
#
#ENV VIRTUAL_ENV=/src
#ENV PATH="$VIRTUAL_ENV/bin:$PATH"

CMD ["python", "-u", "./consumer.py"]