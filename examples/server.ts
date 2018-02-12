import * as express from 'express'
import { Facenet } from '../index'
import * as bodyParser from 'body-parser'
import * as tmp from 'tmp'
import * as fs from 'fs'

const app = express()

app.post('/faces/:index', bodyParser.raw({ type: 'image/jpeg', limit: 5 * 1024 * 1024 }), async (req, res, next) => {
  const index = parseInt(req.params.index, 10)
  const facenet = new Facenet()
  try {
    await saveAsTempFile(req.body, async file => {
      const faceList = await facenet.align(file)
      const buffer = faceList[index].buffer()
      res.writeHead(200)
      res.write(buffer)
      res.end()
    })
  } catch (err) {
    next(err)
  } finally {
    facenet.quit()
  }
})

app.post('/embeddings', bodyParser.raw({ type: 'image/jpeg', limit: 5 * 1024 * 1024 }), async (req, res, next) => {
  const facenet = new Facenet()
  try {
    await saveAsTempFile(req.body, async file => {
      const faceList = await facenet.align(file)
      res.json(await Promise.all(faceList.map(async face => {
        const embedding = await facenet.embedding(face)
        return new Array(embedding.shape[0]).fill(0).map((_, index) => embedding.get(index))
      })))
    })
  } catch (err) {
    next(err)
  } finally {
    facenet.quit()
  }
})

app.listen(8080, (err) => {
  if (err) {
    console.error(err)
  } else {
    console.log('server started on port 8000')
  }
})

async function saveAsTempFile<T>(buffer: Buffer, fn: (file: string) => Promise<T>): Promise<T> {
  const file = await new Promise<string>((resolve, reject) => {
    tmp.file({ prefix: 'facenet-server-temp' }, (err, path) => !err ? resolve(path) : reject(err))
  })
  await new Promise<void>((resolve, reject) => {
    fs.writeFile(file, buffer, err => !err ? resolve() : reject(err))
  })
  return fn(file)
}
