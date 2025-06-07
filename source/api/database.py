import logging
import os
import tempfile
from pickle import dumps, loads

from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import io


POSTGRES_USERNAME = os.environ.get("POSTGRES_USERNAME")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
POSTGRES_HOST = os.environ.get("POSTGRES_HOST")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT")
POSTGRES_DB = os.environ.get("POSTGRES_DB")

# PostgreSQL database connection string
DB_URL = f"postgresql://{POSTGRES_USERNAME}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

Base = declarative_base()


class KerasModel(Base):
    __tablename__ = 'keras_models'
    
    id = Column(Integer, primary_key=True)
    company = Column(String)
    model_data = Column(LargeBinary)
    scaler_data = Column(LargeBinary)
    last_sequence_data = Column(LargeBinary)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    

def init_db():
    # Set up logging
    logging.basicConfig()
    # Log all SQL statements
    # logging.getLogger('sqlalchemy.engine').setLevel(logging.DEBUG)

    engine = create_engine(DB_URL)
    Base.metadata.create_all(engine)
    return engine


def save_model_to_db(model, scaler, last_sequence, company, engine):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp:
        model.save(tmp.name)
        with open(tmp.name, 'rb') as f:
            model_data = f.read()
        os.unlink(tmp.name)

    Session = sessionmaker(bind=engine)
    session = Session()

    scaler_data = dumps(scaler)
    last_sequence_data = dumps(last_sequence)

    keras_model = KerasModel(
        company=company,
        model_data=model_data,
        scaler_data=scaler_data,
        last_sequence_data=last_sequence_data
    )

    try:
        session.add(keras_model)
        session.commit()
        print(f"Model for {company} saved to database successfully")
    except Exception as e:
        session.rollback()
        print(f"Error saving model to database: {e}")
    finally:
        session.close()



def load_model_from_db(company, engine):
    from tensorflow.keras.models import load_model

    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Retorna o ultimo modelo salvo para a empresa
        model_record = session.query(KerasModel) \
            .filter(KerasModel.company == company) \
            .order_by(KerasModel.created_at.desc()) \
            .first()

        print(f"Model: ${model_record}")

        if model_record is None:
            return None

        with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp:
            tmp.write(model_record.model_data)
            tmp.flush()
            model = load_model(tmp.name)
            os.unlink(tmp.name)

        return model, loads(model_record.scaler_data), loads(model_record.last_sequence_data)

    except Exception as e:
        print(f"Erro carregando modelo do banco de dados: {e}")
        return None
    finally:
        session.close()
