
import ctypes, os, sys
os.add_dll_directory(r'C:\Program Files (x86)\CheckerBoard\engines')
os.add_dll_directory(r'C:\Program Files (x86)\CheckerBoard')
dll = ctypes.WinDLL(r'C:\Program Files (x86)\CheckerBoard\engines\Kingsrow64.dll')

CB_FREE = 0; CB_WHITE = 1; CB_BLACK = 2; CB_MAN = 4; CB_KING = 8
BoardType = (ctypes.c_int * 8) * 8

class coor(ctypes.Structure): _fields_ = [('x', ctypes.c_int), ('y', ctypes.c_int)]
class CBmove(ctypes.Structure): _fields_ = [
    ('jumps', ctypes.c_int), ('newpiece', ctypes.c_int), ('oldpiece', ctypes.c_int),
    ('from_sq', coor), ('to_sq', coor),
    ('path', coor * 12), ('delpath', coor * 12), ('delpiece', ctypes.c_int * 12)
]

dll.getmove.argtypes = [BoardType, ctypes.c_int, ctypes.c_double, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.POINTER(CBmove)]
dll.getmove.restype = ctypes.c_int

black_at_top = sys.argv[1] == 'True'
parity = int(sys.argv[2])
color_to_move = int(sys.argv[3])

cb_board = BoardType()
for y in range(8):
    for x in range(8):
        if (x + y) % 2 == parity:
            if y < 3: cb_board[x][y] = (CB_BLACK if black_at_top else CB_WHITE) | CB_MAN
            elif y > 4: cb_board[x][y] = (CB_WHITE if black_at_top else CB_BLACK) | CB_MAN
            else: cb_board[x][y] = CB_FREE
        else: cb_board[x][y] = CB_FREE

# Try disabling WER UI for this process
ctypes.windll.kernel32.SetErrorMode(0x0002)

output = ctypes.create_string_buffer(1024)
playnow = ctypes.c_int(0)
cbmove = CBmove()
res = dll.getmove(cb_board, color_to_move, 1.0, output, ctypes.byref(playnow), 0, 0, ctypes.byref(cbmove))
print(f'{res}|{output.value.decode('utf-8', errors='ignore')}')
