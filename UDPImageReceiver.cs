using UnityEngine;
using UnityEngine.UI;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System;

public class UDPImageReceiver : MonoBehaviour
{
    public RawImage display;
    private Texture2D receivedTexture;
    private UdpClient client;
    private Thread receiveThread;
    private byte[] imageData;
    private object dataLock = new object();
    private bool isRunning = true;

    private const int PORT = 5053;
    private const int MAX_BUFFER = 65535;

    void Start()
    {
        receivedTexture = new Texture2D(2, 2);
        client = new UdpClient(PORT);
        client.Client.ReceiveBufferSize = MAX_BUFFER;
        ///
        client.Client.ReceiveTimeout = 100; // 100 ms
        ///

        receiveThread = new Thread(ReceiveData);
        receiveThread.IsBackground = true;
        receiveThread.Start();
    }

    /*void ReceiveData()
    {
        IPEndPoint remoteEP = new IPEndPoint(IPAddress.Any, PORT);

        while (isRunning)
        {
            try
            {
                byte[] data = client.Receive(ref remoteEP);

                lock (dataLock)
                {
                    imageData = (byte[])data.Clone();
                }
            }
            catch (Exception ex)
            {
                Debug.LogError("UDP Receive Error: " + ex.Message);
            }
        }
    }*/
    void ReceiveData(){
        IPEndPoint remoteEP = new IPEndPoint(IPAddress.Any, PORT);

        while (isRunning)
        {
            try
            {
                byte[] data = client.Receive(ref remoteEP);
                lock (dataLock)
                {
                    imageData = (byte[])data.Clone();
                }
            }
            catch (SocketException ex) when (ex.SocketErrorCode == SocketError.TimedOut)
            {
                // Normal timeout, continuar
            }
            catch (Exception ex)
            {
                Debug.LogError("UDP Receive Error: " + ex.Message);
            }
        }

    }

    void Update()
    {
        byte[] dataCopy = null;

        lock (dataLock)
        {
            if (imageData != null)
            {
                dataCopy = (byte[])imageData.Clone();
                imageData = null;
            }
        }

        if (dataCopy != null)
        {
            bool success = receivedTexture.LoadImage(dataCopy);
            if (success)
            {
                display.texture = receivedTexture;
            }
            else
            {
                Debug.LogWarning("?? Imagen recibida no válida. Revisa tamaño y formato.");
            }
        }
    }

    void OnApplicationQuit()
    {
        isRunning = false;
        receiveThread?.Join();
        client?.Close();
    }
}
