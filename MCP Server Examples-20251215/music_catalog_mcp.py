import sqlite3

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("music_catalog_srv", port=8001)

DB_FILE = 'music.db'

@mcp.tool()
def get_album_by_title(title):
    """
    Returns the album with the given title.

    Args:
        title (str): Title of the album to retrieve.

    Returns:
        tuple: Album data (number, year, album, artist, genre, subgenre, price)
    """
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    query = "SELECT * FROM music WHERE album = ?"
    c.execute(query, (title,))
    result = c.fetchone()
    conn.close()
    return result

@mcp.tool()
def get_albums_by_artist(artist):
    """
    Returns all albums by the given artist.

    Args:
        artist (str): Artist name to retrieve albums for.

    Returns:
        list: List of album data (number, year, album, artist, genre, subgenre, price)
    """
    print("DEBUG: get_albums_by_artist", artist)
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    query = "SELECT * FROM music WHERE artist = ?"
    c.execute(query, (artist,))
    results = c.fetchall()
    conn.close()
    return results

@mcp.tool()
def get_albums_by_year(year):
    """
    Returns all albums released in the given year.

    Args:
        year (int): Year to retrieve albums for.

    Returns:
        list: List of album data (number, year, album, artist, genre, subgenre, price)
    """
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    query = "SELECT * FROM music WHERE year = ?"
    c.execute(query, (year,))
    results = c.fetchall()
    conn.close()
    return results

@mcp.tool()
def get_albums_by_genre(genre):
    """
    Returns all albums of the given genre.

    Args:
        genre (str): Genre to retrieve albums for.

    Returns:
        list: List of album data (number, year, album, artist, genre, subgenre, price)
    """
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    query = "SELECT * FROM music WHERE genre = ?"
    c.execute(query, (genre,))
    results = c.fetchall()
    conn.close()
    return results


def main():
    mcp.run(transport="streamable-http")

if __name__ == "__main__":
    main()
