package atlas.config

import org.chipsalliance.cde.config.Config

class WithAtlas extends Config(
  (site, here, up) => {
    case _ => up(_)
  }
)
