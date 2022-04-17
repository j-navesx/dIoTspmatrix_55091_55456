random = dict({(1,2): 4, (3,4): 6})
random2 = dict({(12,21): 43, (13,414): 64})



gaming = {key: value for key, value in random.items() if key[0] == 1}

print(gaming)

#self._items.get(pos) or self._zero
#random.get((1,3),0)