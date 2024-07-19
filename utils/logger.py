from datetime import datetime
import queue
import threading
from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
import uuid
import json
import os

# 创建声明式基类
Base = declarative_base()


# 定义映射到数据库表的类
class LogEntry(Base):
    __tablename__ = 'logs'
    id = Column(String, primary_key=True)
    label = Column(String)
    msg = Column(Text)
    metadata_info = Column(Text)
    timestamp = Column(DateTime)


# 指定数据库路径
db_path = os.path.expanduser("~/.cache/rag/database/logs.db")

# 确保目录存在
db_directory = os.path.dirname(db_path)
if not os.path.exists(db_directory):
    os.makedirs(db_directory)

# 创建数据库引擎和会话   
engine = create_engine(f'sqlite:///{db_path}')
Session = sessionmaker(bind=engine)

# 确保表存在于数据库中
Base.metadata.create_all(engine)


class Logger(object):
    def __init__(self):
        # 初始化数据库会话
        self.session = Session()
        self.log_queue = queue.Queue()
        self.thread = threading.Thread(target=self._process_logs, daemon=True)
        self.thread.start()

    def _process_logs(self):
        while True:
            try:
                log_entry = self.log_queue.get(timeout=0.5)
                self._log_to_db(log_entry)
                self.log_queue.task_done()
            except queue.Empty:
                continue

    def _log_to_db(self, log_entry):
        new_log = LogEntry(**log_entry)
        self.session.add(new_log)
        self.session.commit()

    def log(self, label, msg, metadata):
        # label: str (table name in the database)
        # msg: str (log message)
        # metadata: dict() {'ip': 'xx.xx.xx.xx', 'input': str, 'output', str, 'xxx': str}
        # t: date time (1e-6 s)

        # 将metadata序列化为JSON格式
        metadata_json = json.dumps(metadata, ensure_ascii=False)
        # 获取当前时间
        t = datetime.now()
        # 创建log_entry对象
        log_entry = {
            'id': str(uuid.uuid4()),
            'label': label,
            'msg': msg,
            'metadata_info': metadata_json,
            'timestamp': t
        }
        self.log_queue.put(log_entry)

    def stop(self):
        self.thread.join()


if __name__ == '__main__':
    print(datetime.now())
    print(type(datetime.now()))
