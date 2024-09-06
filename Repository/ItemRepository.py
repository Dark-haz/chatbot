import json
from sqlalchemy import create_engine , asc, desc , func
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

from Repository.IRepository.IItemRepository import IItemRepository
from Models.Item import Item
from config.static import *

class ItemRepository(IItemRepository):

    def __init__(self) -> None:
        engine = create_engine(DATABASE_URL)
        Session = sessionmaker(bind=engine)
        self.engine = engine
        self.session = Session()

    def find_KNN(self, vector, limit):
        query = text(f'SELECT "itemId","description" FROM "Items" ORDER BY embedding <-> :vector LIMIT :limit')
        
        result = self.session.execute(query, {'vector': str(vector), 'limit': limit}).fetchall()
        
        return result

    def create(self, json_file_path):
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        items = [
            Item(
                itemId=item['itemId'],
                description=item.get('description'),
                embedding=item['embedding']
            )
            for item in data
        ]

        try:
            self.session.bulk_save_objects(items)
            self.session.commit()
            print("Items added successfully.")

        except Exception as e:
            self.session.rollback()
            print(f"An error occurred: {e}")

        finally:
            self.cleanup()

    def cleanup(self):
        if self.session is not None:
            self.session.close()
        if self.engine is not None:
            self.engine.dispose()


