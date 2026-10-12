"""Per-editor local JSONL audit and diagnostics without image contents."""
import json
import logging
import os
from datetime import datetime,timezone
from pathlib import Path
import tempfile
import threading
import traceback
import uuid


class ControlLog:
    def __init__(self):
        base=Path(os.environ.get('LOCALAPPDATA',tempfile.gettempdir()))/'Abu_Align'/'logs'
        self.name='manual_360_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]+'.jsonl'
        self.lock=threading.Lock()
        try:base.mkdir(parents=True,exist_ok=True);self.path=base/self.name;self.path.touch()
        except OSError:self.path=Path(tempfile.gettempdir())/self.name;self.path.touch()
        self.event('editor_open',pid=os.getpid())
    def event(self,event,**details):
        row=dict(time=datetime.now(timezone.utc).isoformat(),event=event,**details)
        with self.lock:
            try:
                with self.path.open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,ensure_ascii=False,default=str)+'\n')
            except OSError:logging.exception('Control-point log write failed')
    def error(self,event,error):self.event(event,error=str(error),traceback=traceback.format_exc())
