import asyncio
import websockets
import json

async def test_websocket():
    async with websockets.connect("ws://127.0.0.1:8001/ws/predict") as websocket:
        data = {
            "features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        }
        await websocket.send(json.dumps(data))  # Envoie les features
        response = await websocket.recv()  # Reçoit la prédiction
        print("Réponse du serveur :", response)

asyncio.run(test_websocket())
