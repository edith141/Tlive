import os
os.environ['dev'] = '1'
from sents import dev_server

# run locally
if __name__ == '__main__':
    dev_server(debug=True)