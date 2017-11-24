from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import scoped_session, sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base

engine = create_engine('sqlite:///mybase.db')

db_session = scoped_session(sessionmaker(bind=engine))

Base = declarative_base()
Base.query = db_session.query_property()


class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    telegram_id = Column(Integer, unique=True)
    first_name = Column(String)
    last_name = Column(String)
    phone = Column(String(120), unique=True)
    avatar = Column(String)
    posts = relationship('Post', backref='author')

    def __repr__(self):
        return '<User {} {}>'.format(self.first_name, self.last_name)


class Post(Base):
    __tablename__ = 'posts'
    id = Column(Integer, primary_key=True)
    created = Column(DateTime)
    content = Column(Text)
    user_id = Column(Integer, ForeignKey('users.telegram_id'))

    def __repr__(self):
        return '<Post {}>'.format(self.content)


if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
