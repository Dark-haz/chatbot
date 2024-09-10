from sqlalchemy import Column, Integer, String, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import ARRAY
from pgvector.sqlalchemy import Vector
from config.static import *

## DOMAIN OBJECT ENTITY MODEL ---------------------------------------------------

Base = declarative_base()      

class Item(Base):
    __tablename__ = 'Items'
    
    # Define columns
    id = Column(Integer, primary_key=True, autoincrement=True)
    itemId = Column(Text, unique=True, nullable=False)
    description = Column(Text, nullable=True)
    embedding =  Column(Vector(VECTOR_DIMENSION)) 

    def to_dict(self):
        return {c.key: getattr(self, c.key) for c in self.__table__.columns}

# Example of how to create the table
if __name__ == "__main__":
    from sqlalchemy import create_engine
    engine = create_engine('postgresql+psycopg2://username:password@localhost:5432/mydatabase')
    Base.metadata.create_all(engine)
