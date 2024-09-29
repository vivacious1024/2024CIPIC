import { app, shell, BrowserWindow, ipcMain } from 'electron'
import { join } from 'path'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import icon from '../../resources/icon.png?asset'
const net = require('net')
let mainWindow = null
let tcpClient = null

function createTcpClient() {
  try {
    // ... 创建和使用 tcpClient 的代码 ...
    tcpClient = net.createConnection({ host: 'localhost', port: 8712 }, () => {
      console.log('Connected to TCP server.')
    })

    tcpClient.on('data', (data) => {
      const message = data.toString()
      console.log('Received data:', message)
      // 将接收到的数据发送到渲染进程
      mainWindow.webContents.send('tcp-data-received', message)
    })

    tcpClient.on('error', (err) => {
      console.error('TCP client error:', err)
    })

    tcpClient.on('close', () => {
      console.log('TCP connection closed')
    })
  } catch (error) {
    console.error('Failed to create TCP client:', error)
  }
}

function createWindow() {
  // Create the browser window.
  mainWindow = new BrowserWindow({
    width: 900,
    height: 670,
    icon: icon,
    show: false,
    autoHideMenuBar: true,
    ...(process.platform === 'linux' ? { icon } : {}),
    webPreferences: {
      //webPreferences：此对象包含用于控制网页行为和特性的选项
      preload: join(__dirname, '../preload/index.js'),
      sandbox: false, //禁用沙盒模式
      contextIsolation: false, //禁用上下文隔离
      nodeIntegration: true //启用 Node.js 集成，使得在渲染进程中可以使用 Node.js 的特性。这对于某些应用来说是必要的，但启用它会带来一定的安全风险。
    }
  })

  //确保了主窗口只有在其内容完全准备好显示时才会显示出来
  mainWindow.on('ready-to-show', () => {
    mainWindow.show()
  })

  //当 Electron 应用试图在新窗口中打开一个链接时，阻止这种行为，并在默认的外部浏览器中打开该链接
  mainWindow.webContents.setWindowOpenHandler((details) => {
    shell.openExternal(details.url)
    return { action: 'deny' }
  })

  // HMR for renderer base on electron-vite cli.
  // Load the remote URL for development or the local html file for production.
  if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.whenReady().then(() => {
  // Set app user model id for windows
  electronApp.setAppUserModelId('com.mmdjiji.stdeeg')

  // Default open or close DevTools by F12 in development
  // and ignore CommandOrControl + R in production.
  // see https://github.com/alex8088/electron-toolkit/tree/master/packages/utils
  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window)
  })

  createWindow()
  createTcpClient()

  app.on('activate', function () {
    // On macOS it's common to re-create a window in the app when the
    // dock icon is clicked and there are no other windows open.
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

// In this file you can include the rest of your app"s specific main process
// code. You can also put them in separate files and require them here.
ipcMain.on('send-tcp-data', (event, data) => {
  if (tcpClient && tcpClient.writable) {
    tcpClient.write(data)
  } else {
    console.error('TCP client is not connected.')
  }
})
