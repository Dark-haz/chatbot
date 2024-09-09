# from data import seeding 

from Item.Repository.ItemRepository import ItemRepository

json_path = "./embeddings_output.json"
item_repository = ItemRepository()

item_repository.create(json_path)