from fastapi import APIRouter, status


router = APIRouter(
    prefix='',
    tags=['check'],
)


@router.get('/healthcheck', status_code=status.HTTP_200_OK)
def perform_healthcheck():
    '''
    It basically sends a GET request to the route & hopes to get a "200"
    response code.
    '''
    return {'healthcheck': 'Everything OK!'}
