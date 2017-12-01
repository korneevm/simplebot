from clarifai.rest import ClarifaiApp, client, Image
import settings


def img_has_cat(filename):
    app = ClarifaiApp(api_key=settings.CLARIFAI_API_KEY)
    model = app.models.get("general-v1.3")
    try:
        image = Image(file_obj=open(filename, 'rb'))
        result = model.predict([image])
        try:
            items = result['outputs'][0]['data']['concepts']
            for item in items:
                if item['name'] == 'cat':
                    return True
            else:
                return False
        except (IndexError):
            return False
    except (client.ApiError, FileNotFoundError):
        return False


if __name__ == "__main__":
    print(img_has_cat("downloads/notcat.jpg"))
    print(img_has_cat("downloads/cat.jpg"))
