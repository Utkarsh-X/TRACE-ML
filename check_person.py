from trace_aml.core.config import load_settings
from trace_aml.store.vector_store import VectorStore

settings = load_settings()
store = VectorStore(settings)
person = store.get_person('PRM001')
if person:
    print('Found PRM001:', person.get('name'))
    print('Profile path:', person.get('profile_photo_path'))
    print('Confidence:', person.get('profile_photo_confidence'))
else:
    print('PRM001 not found!')
