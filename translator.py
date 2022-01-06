from transformers import pipeline

translation_pipeline = pipeline('translation_en_to_de')


def translate_transformers(from_text):
    results = translation_pipeline(from_text)
    return results[0]['translation_text']


if __name__ == '__main__':
    t = input('Write what you want to translate: ')
    r = translation_pipeline(t)
    print('Translation:', r[0]['translation_text'])
