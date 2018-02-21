import requests


for i in range(0, 393):
    response = requests.get('https://s3.amazonaws.com/esports-anal/6789/img_' + str(i) + '_champ_1.png', stream=True).content
    with open('frames/' + str(i) + '.jpg', 'wb') as handler:
        handler.write(response)
