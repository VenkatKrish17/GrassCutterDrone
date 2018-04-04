import socket
import sys
import cv2
import pickle
import numpy as np
import struct ## new

import zstd
#import lz4.frame

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

HOST=''
PORT=8089

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print('Socket created')

s.bind((HOST,PORT))
print('Socket bind complete')
s.listen(True)
print('Socket now listening')

conn,addr=s.accept()

while True:
    length = recvall(conn,16)
    frame_compressed = pickle.loads(recvall(conn, int(length)))
    stringData = zstd.decompress(frame_compressed)
    data = np.fromstring(stringData, dtype='uint8')
    decimg=cv2.imdecode(data,1)
    cv2.imshow('server',decimg)
    if cv2.waitKey(25) & 0xFF==ord('q'):
        break
cv2.destroyAllWindows()
cap.release()
s.close()
