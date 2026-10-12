"""Small monochrome line icons, rendered locally without font/emoji dependencies."""
from PIL import Image, ImageDraw, ImageTk


def icon_name(text):
    value = text.lower().strip()
    if value.startswith(('open ', 'browse')) and 'log' not in value:
        return 'folder'
    for words, name in [
        (('alignment','align','fit','focus','reconstruction'), 'target'),
        (('training','train','start','run','resume','extract'), 'play'),
        (('stop','cancel'), 'stop'), (('pause',), 'pause'),
        (('folder','browse','dataset','output','results','open run'), 'folder'),
        (('photo','camera','inspect'), 'camera'),
        (('compare','switch'), 'compare'), (('refresh','reload','reset','restore'), 'refresh'),
        (('save','profiles','preset'), 'save'),
        (('find','search'), 'search'), (('activity','monitor','hardware'), 'activity'),
        (('guide','report','log','help'), 'document'),
        (('3d','rig','workspace'), 'cube'),
        (('settings','options','quality','sampling','optics'), 'sliders')]:
        if any(word in value for word in words):
            return name
    return None


def line_icon(widget, name, color='#a0adb5', size=16):
    root = widget._root()
    cache = getattr(root, '_line_icons', None)
    if cache is None:
        cache = root._line_icons = {}
    key = (name, color, size)
    if key in cache:
        return cache[key]
    image = Image.new('RGBA', (64,64))
    d = ImageDraw.Draw(image)
    def line(points):
        d.line([(x*4,y*4) for x,y in points], fill=color, width=5, joint='curve')
    def box(bounds):
        d.rectangle(tuple(v*4 for v in bounds), outline=color, width=5)
    if name == 'target':
        d.ellipse((12,12,52,52),outline=color,width=5)
        for pts in [[(8,1),(8,5)],[(8,11),(8,15)],[(1,8),(5,8)],[(11,8),(15,8)]]: line(pts)
    elif name == 'play': line([(5,3),(12,8),(5,13),(5,3)])
    elif name == 'stop': box((4,4,12,12))
    elif name == 'pause': line([(5,3),(5,13)]);line([(11,3),(11,13)])
    elif name == 'folder': line([(2,13),(2,3),(6,3),(8,5),(14,5),(14,13),(2,13)])
    elif name == 'camera':
        line([(2,12),(2,5),(5,5),(6,3),(10,3),(11,5),(14,5),(14,12),(2,12)])
        d.ellipse((24,24,40,40),outline=color,width=5)
    elif name == 'compare':
        box((2,3,6,13));box((10,3,14,13))
    elif name == 'refresh':
        d.arc((12,12,52,52),40,315,fill=color,width=5)
        line([(10,2),(13,4),(14,1)])
    elif name == 'save':
        box((3,2,13,14));box((6,2,10,6));box((6,10,10,14))
    elif name == 'search':
        d.ellipse((8,8,40,40),outline=color,width=5);line([(10,10),(14,14)])
    elif name == 'activity': line([(1,9),(4,9),(6,3),(9,13),(11,7),(15,7)])
    elif name == 'document':
        box((4,2,12,14));line([(6,6),(10,6)]);line([(6,9),(10,9)])
    elif name == 'cube':
        line([(8,1),(14,4),(14,12),(8,15),(2,12),(2,4),(8,1)])
        line([(2,4),(8,7),(14,4)]);line([(8,7),(8,15)])
    elif name == 'sliders':
        for y,x in [(4,5),(8,11),(12,7)]:
            line([(2,y),(14,y)]);box((x-1,y-1,x+1,y+1))
    icon = ImageTk.PhotoImage(image.resize((size,size),Image.Resampling.LANCZOS),master=root)
    cache[key] = icon
    return icon
