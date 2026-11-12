import sys, asyncio, subprocess, os
from playwright.async_api import async_playwright
start,end,out=int(sys.argv[1]),int(sys.argv[2]),sys.argv[3]
tmp=out.replace('.mp4','.tmp.mp4')
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(executable_path=os.environ.get('CHROME','/usr/bin/google-chrome'),args=['--no-sandbox','--hide-scrollbars','--force-color-profile=srgb'])
        pg=await b.new_page(viewport={'width':1920,'height':1080})
        await pg.goto('file://'+os.path.abspath(os.path.join(os.path.dirname(__file__),'index.html')))
        await pg.evaluate('document.fonts.ready')
        ff=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','image2pipe','-framerate','30','-c:v','mjpeg','-i','-',
          '-vf','scale=in_range=full:in_color_matrix=bt601:out_range=tv:out_color_matrix=bt709,format=yuv420p',
          '-c:v','libx264','-preset','veryfast','-crf','16','-r','30','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709','-movflags','+faststart',tmp],stdin=subprocess.PIPE)
        for f in range(start,end):
            await pg.evaluate(f'render({f/30})')
            ff.stdin.write(await pg.screenshot(type='jpeg',quality=92))
        ff.stdin.close(); rc=ff.wait(); await b.close()
        if rc==0: os.rename(tmp,out)
        else: sys.exit(rc)
asyncio.run(main())
