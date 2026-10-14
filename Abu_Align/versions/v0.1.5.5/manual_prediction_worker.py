"""Isolated inference/setup worker. Network is disabled during image inference."""
import argparse,json,sys
from pathlib import Path


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--request');parser.add_argument('--setup',action='store_true');args=parser.parse_args()
    def emit(event):print('ABU_PREDICTION '+json.dumps(event),flush=True)
    try:
        from manual_prediction_engine import setup_models,scan
        if args.setup:emit(dict(event='setup_complete',models=setup_models()))
        else:
            # Enforce no outbound sockets in the inference process, including
            # accidental pretrained-loader/telemetry calls by dependencies.
            import socket
            def offline(*a,**k):raise RuntimeError('Network access is disabled during local prediction.')
            socket.create_connection=offline;socket.socket.connect=offline;socket.socket.connect_ex=offline
            request=json.loads(Path(args.request).read_text(encoding='utf-8'));scan(request,emit)
    except Exception as error:emit(dict(event='error',error=str(error)));sys.exit(1)
